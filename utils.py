from absl import logging
import numpy as np
from PIL import Image
import torch
from torch.distributions.categorical import Categorical
from rdkit import RDLogger
import re
RDLogger.DisableLog('rdApp.*')


@torch.no_grad()
def AE_SMILES_encoder(sm, ae_model):
    if sm[0][:5] != "[CLS]":
        sm = ["[CLS]"+s for s in sm]
    text_input = ae_model.tokenizer(sm, padding='max_length', truncation=True, max_length=128, return_tensors="pt").to(ae_model.device)
    text_input_ids = text_input['input_ids'][:, 1:]
    text_attention_mask = text_input['attention_mask'][:, 1:]
    if hasattr(ae_model.text_encoder2, 'bert'):
        output = ae_model.text_encoder2.bert(text_input_ids, attention_mask=text_attention_mask, return_dict=True, mode='text').last_hidden_state
    else:
        output = ae_model.text_encoder2(text_input_ids, attention_mask=text_attention_mask, return_dict=True).last_hidden_state

    if hasattr(ae_model, 'encode_prefix'):
        output = ae_model.encode_prefix(output)
        if ae_model.output_dim*2 == output.size(-1):
            mean, logvar = torch.chunk(output, 2, dim=-1)
            logvar = torch.clamp(logvar, -30.0, 20.0)
            std = torch.exp(0.5 * logvar)
            output = mean + std * torch.randn_like(mean)
    return output


@torch.no_grad()
def generate(model, image_embeds, text, stochastic=True, prop_att_mask=None, k=None):
    text_atts = torch.where(text == 0, 0, 1)
    if prop_att_mask is None:   prop_att_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image_embeds.device)
    token_output = model.text_encoder(text,
                                      attention_mask=text_atts,
                                      encoder_hidden_states=image_embeds,
                                      encoder_attention_mask=prop_att_mask,
                                      return_dict=True,
                                      is_decoder=True,
                                      return_logits=True,
                                      )[:, -1, :]  # batch*300
    if k:
        p = torch.softmax(token_output, dim=-1)
        if stochastic:
            output = torch.multinomial(p, num_samples=k, replacement=False)
            return torch.log(torch.stack([p[i][output[i]] for i in range(output.size(0))])), output
        else:
            output = torch.topk(p, k=k, dim=-1)  # batch*k
            return torch.log(output.values), output.indices
    if stochastic:
        p = torch.softmax(token_output, dim=-1)
        m = Categorical(p)
        token_output = m.sample()
    else:
        token_output = torch.argmax(token_output, dim=-1)
    return token_output.unsqueeze(1)  # batch*1


@torch.no_grad()
def AE_SMILES_decoder(pv, model, stochastic=False, k=2, max_length=150):
    if hasattr(model, 'decode_prefix'):
        pv = model.decode_prefix(pv)

    tokenizer = model.tokenizer
    if tokenizer is None:
        raise ValueError('Tokenizer is not defined')
    # test
    model.eval()
    candidate = []
    if k == 1:
        text_input = torch.tensor([tokenizer.cls_token_id]).expand(pv.size(0), 1).to(model.device)  # batch*1
        for _ in range(max_length):
            output = generate(model, pv, text_input, stochastic=False)
            if output.sum() == 0:
                break
            text_input = torch.cat([text_input, output], dim=-1)
        for i in range(text_input.size(0)):
            sentence = text_input[i]
            if tokenizer.sep_token_id in sentence: sentence = sentence[:(sentence == tokenizer.sep_token_id).nonzero(as_tuple=True)[0][0].item()]
            cdd = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(sentence)).replace(' ', '').replace('[CLS]', '')
            candidate.append(cdd)
    else:
        for prop_embeds in pv:
            prop_embeds = prop_embeds.unsqueeze(0)
            product_input = torch.tensor([tokenizer.cls_token_id]).expand(1, 1).to(model.device)
            values, indices = generate(model, prop_embeds, product_input, stochastic=stochastic, k=k)
            product_input = torch.cat([torch.tensor([tokenizer.cls_token_id]).expand(k, 1).to(model.device), indices.squeeze(0).unsqueeze(-1)], dim=-1)
            current_p = values.squeeze(0)
            final_output = []
            for _ in range(max_length):
                values, indices = generate(model, prop_embeds, product_input, stochastic=stochastic, k=k)
                k2_p = current_p[:, None] + values
                product_input_k2 = torch.cat([product_input.unsqueeze(1).repeat(1, k, 1), indices.unsqueeze(-1)], dim=-1)
                if tokenizer.sep_token_id in indices:
                    ends = (indices == tokenizer.sep_token_id).nonzero(as_tuple=False)
                    for e in ends:
                        p = k2_p[e[0], e[1]].cpu().item()
                        final_output.append((p, product_input_k2[e[0], e[1]]))
                        k2_p[e[0], e[1]] = -1e5
                    if len(final_output) >= k ** 1:
                        break
                current_p, i = torch.topk(k2_p.flatten(), k)
                next_indices = torch.from_numpy(np.array(np.unravel_index(i.cpu().numpy(), k2_p.shape))).T
                product_input = torch.stack([product_input_k2[i[0], i[1]] for i in next_indices], dim=0)

            candidate_k = []
            final_output = sorted(final_output, key=lambda x: x[0], reverse=True)[:k]
            for p, sentence in final_output:
                cdd = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(sentence[:-1])).replace('[CLS]', '')
                candidate_k.append(cdd)
            if candidate_k == []:
                candidate.append("")
            else:
                candidate.append(candidate_k[0])
            # candidate.append(random.choice(candidate_k))
    return candidate


@torch.no_grad()
def molT5_encoder(descriptions, molt5, molt5_tokenizer, description_length, device):
    tokenized = molt5_tokenizer(descriptions, padding='max_length', truncation=True, max_length=description_length, return_tensors="pt").to(device)
    encoder_outputs = molt5.encoder(input_ids=tokenized.input_ids, attention_mask=tokenized.attention_mask, return_dict=True).last_hidden_state
    return encoder_outputs, tokenized.attention_mask


def get_validity(smiles):
    from rdkit import Chem
    v = []
    for l in smiles:
        try:
            if l == "":
                continue
            s = Chem.MolToSmiles(Chem.MolFromSmiles(l), isomericSmiles=False)
            v.append(s)
        except:
            continue
    u = list(set(v))
    if len(u) == 0:
        return 0., 0.
    return len(v) / len(smiles)


alphabets = "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov|edu|me)"
digits = "([0-9])"
multiple_dots = r'\.{2,}'


def split_into_sentences(text: str) -> list[str]:
    """
    Split the text into sentences.

    If the text contains substrings "<prd>" or "<stop>", they would lead
    to incorrect splitting because they are used as markers for splitting.

    :param text: text to be split into sentences
    :type text: str

    :return: list of sentences
    :rtype: list[str]
    """
    text = " " + text + "  "
    text = text.replace("\n", " ")
    text = re.sub(prefixes, "\\1<prd>", text)
    text = re.sub(websites, "<prd>\\1", text)
    text = re.sub(digits + "[.]" + digits, "\\1<prd>\\2", text)
    text = re.sub(multiple_dots, lambda match: "<prd>" * len(match.group(0)) + "<stop>", text)
    if "Ph.D" in text: text = text.replace("Ph.D.", "Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] ", " \\1<prd> ", text)
    text = re.sub(acronyms + " " + starters, "\\1<stop> \\2", text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]", "\\1<prd>\\2<prd>\\3<prd>", text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]", "\\1<prd>\\2<prd>", text)
    text = re.sub(" " + suffixes + "[.] " + starters, " \\1<stop> \\2", text)
    text = re.sub(" " + suffixes + "[.]", " \\1<prd>", text)
    text = re.sub(" " + alphabets + "[.]", " \\1<prd>", text)
    if "”" in text: text = text.replace(".”", "”.")
    if "\"" in text: text = text.replace(".\"", "\".")
    if "!" in text: text = text.replace("!\"", "\"!")
    if "?" in text: text = text.replace("?\"", "\"?")
    text = text.replace(".", ".<stop>")
    text = text.replace("?", "?<stop>")
    text = text.replace("!", "!<stop>")
    text = text.replace("<prd>", ".")
    sentences = text.split("<stop>")
    sentences = [s.strip() for s in sentences]
    if sentences and not sentences[-1]: sentences = sentences[:-1]
    return sentences


def center_crop(width, height, img):
    resample = {'box': Image.BOX, 'lanczos': Image.LANCZOS}['lanczos']
    crop = np.min(img.shape[:2])
    img = img[(img.shape[0] - crop) // 2: (img.shape[0] + crop) // 2, (img.shape[1] - crop) // 2: (img.shape[1] + crop) // 2]  # center crop
    try:
        img = Image.fromarray(img, 'RGB')
    except:
        img = Image.fromarray(img)
    img = img.resize((width, height), resample)  # resize the center crop from [crop, crop] to [width, height]

    return np.array(img).astype(np.uint8)


def set_logger(log_level='info', fname=None):
    import logging as _logging
    handler = logging.get_absl_handler()
    formatter = _logging.Formatter('%(asctime)s - %(filename)s - %(message)s')
    handler.setFormatter(formatter)
    logging.set_verbosity(log_level)
    if fname is not None:
        handler = _logging.FileHandler(fname)
        handler.setFormatter(formatter)
        logging.get_absl_logger().addHandler(handler)


def drawRoundRec(draw, color, x, y, w, h, r):
    drawObject = draw

    '''Rounds'''
    drawObject.ellipse((x, y, x + r, y + r), fill=color)
    drawObject.ellipse((x + w - r, y, x + w, y + r), fill=color)
    drawObject.ellipse((x, y + h - r, x + r, y + h), fill=color)
    drawObject.ellipse((x + w - r, y + h - r, x + w, y + h), fill=color)

    '''rec.s'''
    drawObject.rectangle((x + r / 2, y, x + w - (r / 2), y + h), fill=color)
    drawObject.rectangle((x, y + r / 2, x + w, y + h - (r / 2)), fill=color)
