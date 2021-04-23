import torch
from modeling_nezha import BertPreTrainedModel, BertEmbeddings, BertEncoder, BertOnlyMLMHead

from bojone_snippets import AutoRegressiveDecoder
from bojone_tokenizers import load_vocab, Tokenizer
from configuration.config import *


dict_path = str(cdial_gpt_nezha_pt_path / "vocab.txt")

# 加载并精简词表
token_dict, keep_tokens = load_vocab(
    dict_path=dict_path,
    simplified=True,
    startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
)

tokenizer = Tokenizer(token_dict, do_lower_case=True)


class BertModel(BertPreTrainedModel):
    def __init__(self, config):
        super(BertModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.apply(self.init_bert_weights)

    def compute_attention_bias(self, segment_ids):
        seq_len = segment_ids.size(1)
        idxs = torch.arange(0, seq_len)
        mask = idxs[None, :] <= idxs[:, None]
        mask = mask.to(torch.float).to(segment_ids.device)
        return -(1 - mask[None, None]) * 1e12

    def forward(self, input_ids, token_type_ids):
        extended_attention_mask = self.compute_attention_bias(token_type_ids)

        embedding_output = self.embeddings(input_ids=input_ids, token_type_ids=token_type_ids)

        padding_mask = (input_ids != 0).to(input_ids.device)

        encoder_layers,_ = self.encoder(embedding_output, attention_mask=extended_attention_mask,
                                      head_mask=padding_mask)
        sequence_output = encoder_layers[-1]
        return sequence_output


class DialogModel(BertPreTrainedModel):
    def __init__(self, config):
        super(DialogModel, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config, self.bert.embeddings.word_embeddings.weight)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids):
        sequence_output = self.bert(input_ids, token_type_ids)

        prediction_scores = self.cls(sequence_output)  # [b,s,V]

        return prediction_scores


model = DialogModel.from_pretrained(pretrained_model_name_or_path=cdial_gpt_nezha_pt_path)

model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

class ChatBot(AutoRegressiveDecoder):
    """基于随机采样对话机器人
    """
    @AutoRegressiveDecoder.wraps(default_rtype='probas')
    def predict(self, inputs, output_ids, states):
        token_ids, segment_ids = inputs
        token_ids = np.concatenate([token_ids, output_ids], 1)
        curr_segment_ids = np.ones_like(output_ids) - segment_ids[0, -1]
        segment_ids = np.concatenate([segment_ids, curr_segment_ids], 1)

        token_ids = torch.tensor(token_ids, dtype=torch.long, device=device)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long, device=device)

        with torch.no_grad():
            pred = model(token_ids, segment_ids)
            pred = torch.softmax(pred, dim=-1)
            pred = pred[:, -1].cpu().detach().numpy()

        return pred

    def response(self, texts, topk=5):
        token_ids, segment_ids = [tokenizer._token_start_id], [0]
        for i, text in enumerate(texts):
            ids = tokenizer.encode(text)[0][1:]
            token_ids.extend(ids)
            segment_ids.extend([i % 2] * len(ids))
        results = self.random_sample([token_ids, segment_ids], 1, topk)
        return tokenizer.decode(results[0])


chatbot = ChatBot(start_id=None, end_id=tokenizer._token_end_id, maxlen=32)

print(chatbot.response(['别爱我没结果', '你这样会失去我的', '失去了又能怎样']))
print(chatbot.response(["我今天腿都废了，你们过节，我搬砖", "辛苦啊，圣诞节还去赚大钱了加油"]))
print(chatbot.response(["这个会不会聚划算","暂时没有哦","后期会不会有"]))
print(chatbot.response(["前排，鲁迷们都起床了吧","标题说助攻，但是看了那球，真是活生生的讽刺了"]))
print(chatbot.response(["火锅我在重庆成都吃了七八顿火锅","哈哈哈哈！那我的嘴巴 可能要烂掉！"]))
print()
print(chatbot.response(["在","在的亲有什么可以帮您吗","还送不送的不送货我退了","亲我们这边帮您更换其他快递了","31号到今天3天了","亲真是不好意思呢我这边再帮您催一下快递","今天不到退了什么鬼快递","真是对不起了亲","31号中午就到这边了就是不送","好的亲我这边帮您向邮政那边反应了","那我退货算了事不过三你说对吧","亲最近是元旦呢可能物流量大我明天帮您和EMS联系帮您确认一下具体情况您看可以吗","今天已经3天了","亲您都等了三天了呢我明天帮您和快递联系一下知道具体情况后您再决定可以吗不然您这3天都白等了呢","有什么办法快递都送给我了我还是去实体店买吧心累","实在是对不起您了那如果后续快递再给您送来的话麻烦您拒收一下就可以退回了亲","你点下退款吧","您稍等相应的工作人员会帮您处理的","快递不送有什么办法","好的亲","已经是马上到手的东西叫你等3天你自己想想电话都没一个来安慰一下我","真对不起了亲您的情况我会帮您记录的您下次再来我们店我们给您送点额外的小礼品","你自己说对不对","是的亲","我这次第二次在你这里买了感觉好才买谁知道快递不给力","下次麻烦您提醒一下我们给您换其他的快递","哦你退款了我重新下单","好的亲退款是相应的工作人员受理的麻烦您等待一下就可以了","好","麻烦您了亲"]))



