#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/11/9 下午3:04
# @Author  : Shihan Ran
# @Site    : 
# @File    : thoughts.py
# @Software: PyCharm

# this file is aimed to process sentence to word
# and use sentiment dict to score a sentence

import numpy
import jieba

stop_words = [w.strip() for w in open('./dict/notWord.txt', 'r', encoding='GBK').readlines()]
stop_words.extend(['\n','\t',' '])

def Sent2Word(sentence):
   # 去除停用词
   global stop_words

   words = jieba.cut(sentence)
   words = [w for w in words if w not in stop_words]

   return words

# 加载各种词典
def LoadDict():
   # 情感词
   pos_words = open('./dict/pos_word.txt').readlines()
   pos_dict = {}
   for w in pos_words:
      word, score= w.strip().split(',')
      pos_dict[word] = float(score)

   neg_words = open('./dict/neg_word.txt').readlines()
   neg_dict = {}
   for w in neg_words:
      word, score = w.strip().split(',')
      neg_dict[word] = float(score)

   # 否定词 ['不', '没', '无', '非', '莫', '弗', '勿', '毋', '未', '否', '别', '無', '休', '难道']
   not_words = open('./dict/notDict.txt').readlines()
   not_dict = {}
   for w in not_words:
      word = w.strip()
      not_dict[word] = float(-1)

   #程度副词 {'百分之百': 10.0, '倍加': 10.0, ...}
   degree_words = open('./dict/degreeDict.txt').readlines()
   degree_dict = {}
   for w in degree_words:
       word,score = w.strip().split(',')
       degree_dict[word] = float(score)

   return pos_dict, neg_dict, not_dict, degree_dict

# 定位句子中的 情感词 否定词 程度副词的 【位置】
def LocateSpecialWord(pos_dict, neg_dict, not_dict, degree_dict, sent):
   pos_word = {}
   neg_word = {}
   not_word = {}
   degree_word = {}

   for index, word in enumerate(sent):
      if word in pos_dict:
         pos_word[index] = pos_dict[word]
         print('pos', word)
      elif word in neg_dict:
         neg_word[index] = neg_dict[word]
         print('neg', word)
      elif word in not_dict:
         not_word[index] = -1
         print('not', word)
      elif word in degree_dict:
         degree_word[index] = degree_dict[word]
         print('degree', word)

   return pos_word, neg_word, not_word, degree_word

# 计算该句子的分数
def ScoreSent(pos_word, neg_word, not_word, degree_word, words):
   W=1
   score=0

   # 存所有情感词的位置的列表
   pos_locs = list(pos_word.keys())
   neg_locs = list(neg_word.keys())
   not_locs = list(not_word.keys())
   degree_locs = list(degree_word.keys())

   posloc=-1  # 已检测到多少词
   negloc=-1

   # 遍历句中所有单词words，i为单词绝对位置
   for i in range(0,len(words)):
       # 如果该词为积极词
       if i in pos_locs:
           # loc为情感词位置列表的序号
           posloc += 1
           # 直接添加该情感词分数
           score += W * float(pos_word[i])

           if posloc < len(pos_locs)-1:
               # 判断该情感词与下一情感词之间是否有否定词或程度副词
               # j为绝对位置
               # print(pos_locs)
               for j in range(pos_locs[posloc], pos_locs[posloc+1]):
                   # 如果有否定词
                   if j in not_locs:
                      W *= -1
                   # 如果有程度副词
                   elif j in degree_locs:
                      # print('degree', words[j])
                      W *= degree_word[j]
           #         # else:
           #         #     W *= 1

       elif i in neg_locs:
          # loc为情感词位置列表的序号
          negloc += 1
          # 直接添加该情感词分数
          # print(i)
          # print(score, W * float(neg_word[i]))
          score += (-1) * W * float(neg_word[i])

          if negloc < len(neg_locs) - 1:
             # 判断该情感词与下一情感词之间是否有否定词或程度副词
             # j为绝对位置
             for j in range(neg_locs[negloc], neg_locs[negloc + 1]):
                # 如果有否定词
                if j in not_locs:
                   # print('not', words[j])
                   W *= -1
                # 如果有程度副词
                elif j in degree_locs:
                   # print('degree', words[j])
                   W *= degree_word[j]
          #       # else:
          #       #     W = 1

   # print(numpy.sign(score)*numpy.log(abs(score)))
   return score

# pos_dict, neg_dict, not_dict, degree_dict = LoadDict()
#
# title = '据《中国航天报》报道，中国航天科技集团集团公司董事长、党组书记雷凡培日前在集团总部述职会上透露，当前集团公司正在按中央要求 深化国有企业改革、军工科研事业单位改制和规范董事会建设，积极推进经济结构转型和管控模式调整；在型号工作方面，面临着体系化发展、高强度研制和高密度 发射、竞争性发展、跨领域融合创新发展以及国际化发展的新常态。\n从盘面看，军工板块今日井喷。其 中，航天科技集团旗下的航天机电涨停，其他军工股如中国卫星、航天动力、航天电子、中航飞机、中航机电等中航系军工股也强势大涨。市场人士分析，此轮上涨 一方面是受益于军工企业管理层近期对于改革转型较为积极的表态，另一方面也是因为军工板块在牛市行情下上涨势能的爆发。\n银河证券指出，日前有媒体报道军工科研院所改革分类方案确定，全军武器装备采购信息网、国家军民结合公共服务平台等重要平台正式上线运行，持续建议超配军工 板块。中航工业防务板块整合启动，军工集团资本运作和军工科研院所改制预期提升，总装鼓励民企参与武器装备研发，中长期看好国防军工板块。尤其是军工装备 类的标的股票，包括空军装备、通讯导航装备、海军装备、兵器装备。\n华创证券则认为，2015年军工 行业科研院所改制进程明显加快，回顾之前表现，航天科技集团和航天科工集团的改革意图较明显，这两大集团的改制步伐走在了前列。同时，这两大集团资产优 质、科研院所众多，未来的改革或超预期。此外，新一轮军队建设核心是信息化，信息化建设的重大科技项目上马是大概率事件。\n有国资改革专家认为，2015年，国企改革的重点任务逐步明确，“加减乘除”调结构，建立“三项清单”和“四个一批”，以分类、混合所有制以及“四项试点” 改革为重点加速推进。分类改革的思路已经清晰，也发出大规模重组的信号。2015年，在分类改革完成的基础上，企业必然会迎来大分化、大调整、大改组的过 程，由国资委监管的112户央企很可能会重组为30至50户左右。\n在国企改革的投资机会中，行业特 征较为明显的是军工。我国军工企业证券化率比较低，股权结构简单，十大军工集团目前的证券化率仅有31%，而国外军工企业资产证券化率约为 70%~80%，相比之下我国军工企业潜在资产证券化的空间较大，相关上市公司有望成为资产证券化的重要运作平台。在十大军工集团中， 从改革进度、资产证券化空间、资产质量以及相关上市标的等方面考量，中航工业集团、中国航天科技集团、中国航天科工集团、中国电子科技集团、中国兵工集 团、中国兵器装备集团等六大军工集团旗下上市公司具有更大投资机会。其中，中国电科的整合最值得期待，相比航天系集团，中国电科的资产分散，整合潜力巨 大，下属55家研究院，目前已经上市的海康威视等7家公司分别隶属7家不同的研究所，资产证券化率刚刚超过20%，是军工集团中相对偏低的。上述军工集团 的混改和资产证券化值得期待。\n泰豪科技（600590）已制定以智能电力及军工 装备为主业的发展战略。泰豪军工产业专业从事通信指挥系统、军用电站、卫星导航、弹药引信、雷达、特种空调等产品的科研、生产与服务，产品装备到陆、海、 空、二炮等各军兵种，广泛应用于雷达、火炮、导弹、通信指挥、电子对抗等各种武器装备系统，具备一定特殊垄断性。此前公司发布定增方案拟募资11亿元，其 中天津硅谷天堂、新疆硅谷天堂各认购992万股，增添了军工装备并购重组预期。\n中国一重（601106）主 要为钢铁、有色、电力等行业及国防军工提供重大成套技术装备、高新技术产品和服务，并开展相关的国际贸易。现已形成以核电、水电、风电成套设备及煤化工设 备、石油开采与加工设备为代表的能源装备，以冶金成套设备等为代表的工业装备，以核电铸锻件、火电铸锻件、船用铸锻件等为代表的装备基础材料等四大产业板 块。\n中航重机（600765）隶属中国航空工业集团，是中国航空工业企业首家上 市公司，被誉为“中国航空工业第一股”。公司以航空技术为基础，建立了锻铸、液压、新能源投资三大业务发展平台，积极发展高端宇航锻铸造业务、高端液压系 统业务、高端散热系统业务、中小型燃机成套业务，燃机成套向总承包、安装、运行维护等服务领域拓展。公司产品大量应用于国内外航空航天、新能源、工程机械 等领域，成为了中国最具竞争力的高端装备制造企业。\n中航黑豹（600760）是 原国家机械局定点生产低速货车的骨干企业，具备年产10万辆整车的综合生产能力。专设黑豹技术开发中心，能承担高水准低速货车、汽车、皮卡车等产品的研究 和开发。目前公司产品销售网络已遍布全国各省、市、自治区，并拥有销售形式看好的埃及、叙利亚、越南、中美洲等十几个国家的国际市场。\xa0\xa0 \xa0\n财联社声明：文章内容仅供参考，不构成投资建议。投资者据此操作，风险自担。'
# title = Sent2Word(title)
#
# pos_word, neg_word, not_word, degree_word = \
#             LocateSpecialWord(pos_dict, neg_dict, not_dict, degree_dict, title)
# print(pos_word, neg_word, not_word, degree_word)
# print(ScoreSent(pos_word, neg_word, not_word, degree_word, title))