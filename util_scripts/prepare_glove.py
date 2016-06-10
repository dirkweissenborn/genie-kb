import tensorflow as tf
import web.embeddings
import web.embedding
import os

#tf.app.flags.DEFINE_string('embedding_format', 'glove', 'glove|word2vec_bin')
tf.app.flags.DEFINE_integer('dim', 100, 'embedding dimensionality')
tf.app.flags.DEFINE_string('corpus', 'wiki-6B', """Glove training corpus.
        * wiki-6B: 50, 100, 200, 300
        * common-crawl-42B: 300
        * common-crawl-840B: 300
        * twitter: 25, 50, 100, 200""")
tf.app.flags.DEFINE_string('out', None, 'path to output embeddings')

FLAGS = tf.app.flags.FLAGS

if FLAGS.out is None:
    FLAGS.out = "embeddings/glove_%s_%d.bin" % (FLAGS.corpus, FLAGS.dim)

print("Fetching & loading Glove and saving to ~/web_data/embeddings/ ...")
e = web.embeddings.fetch_GloVe(dim=FLAGS.dim, corpus=FLAGS.corpus, normalize=True, lower=False)

os.makedirs("embeddings", exist_ok=True)
print("Saving as binary in word2vec format to %s..." % FLAGS.out)
web.embedding.Embedding.to_word2vec(e, FLAGS.out, binary=True)
