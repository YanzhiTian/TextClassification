import config
import loadCorpus

if __name__ == '__main__':
    # 将语料库中读取的单词按索引顺序存储在文本中，包括了<pad>,<num>,<unk>词标
    # 由于应用中只需要使用词典，并不需要读取语料库中的内容，因此这样处理能让应用打开速度更快
    cfg = config.config()
    lc = loadCorpus.loadCorpus(cfg.vocab_path, cfg.corpus_path)
    with open(cfg.save_dict_path, 'w', encoding='utf-8') as f:
        for word in lc.word2index:
            f.write(word+'\n')
    f.close()
