curl --create-dirs --output models/wikitext/model https://dl.fbaipublicfiles.com/fairseq/models/wiki103_fconv_lm.tar.bz2
tar xjf models/wikitext/model -C models/wikitext
rm models/wikitext/model