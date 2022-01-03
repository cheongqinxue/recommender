import streamlit as st
import faiss
from faiss import METRIC_INNER_PRODUCT
import pandas as pd
import joblib
import numpy as np
import torch
from bipartite_models import TransRBipartiteModel
import requests
import s3fs
fs = s3fs.S3FileSystem(anon=False)

class Args:
    datapath = 's3://qx-poc-public/recommender/'
    modelpath = 's3://qx-poc-public/recommender/transrBipartite-marginloss0_5-800epoch-5neg/'
        
@st.cache(ttl=600)
def load_joblib(path):
    with fs.open(path) as f:
        data = joblib.load(f)
    return data

@st.cache(ttl=600)
def load_np(path):
    with fs.open(path) as f:
        data = np.load(f)
    return data

def load(path):
    df = load_joblib(path+'/df.joblib')
    df.published_date = pd.to_datetime(df.published_date, errors='coerce', unit='s')
    emb = load_np(path+'/embeds.np.npy')
    return df, emb

@st.cache(ttl=600)
def model_fn(path):
    model, head2ix = TransRBipartiteModel.load_s3_pretrained(args.modelpath)
    return model, head2ix

def criteria(idx, col, fn):
    try:
        x = ds[idx]
        return fn(x[col])
    except:
        return False

def search(domain, rep_vectors, faiss_index, df, head2ix, embeddings, model, display_top_n=20, 
    search_n_per_signpost=5000, language='any', debug=False, favor='na'):
    favor = favor.split(',')
    if all([sn.isnumeric() for sn in favor]):
        favor = [int(sn) for sn in favor]
        _, scores, indices = faiss_index.range_search(embeddings[favor,:], 0.55)
    else:
        scores, indices = faiss_index.search(torch.vstack(rep_vectors['rep_vectors'][domain]).numpy(), 
            search_n_per_signpost)
    indices = list(set(indices.reshape(-1).tolist()))

    with torch.no_grad():
        h = head2ix[domain]
        te = torch.tensor(embeddings[indices], device='cpu')
        scores = model.scoring_function(
                h_idx=torch.tensor([h], device = 'cpu'),
                r_idx=torch.tensor([0], device = 'cpu'),
                t_idx=None,
                new_tails=te)
        scores = torch.tanh(scores+2.5)
        topn = torch.argsort(scores, descending=True)[:max(300, int(search_n_per_signpost/4))].tolist()

    indices_ = np.asarray(indices)[topn].tolist()
    scores_ = scores[topn].numpy().tolist()
    resultdf = df.iloc[indices_].drop(columns=['media_item_id','published_date'])
    resultdf['score'] = scores_
    resultdf = resultdf.drop_duplicates(subset='title')
    if language != 'any':
        resultdf = resultdf[resultdf.language==language]
    resultdf = resultdf.drop(columns=['language'])
    try:
        return resultdf.head(display_top_n)
    except Exception as e:
        print('topn ', topn[:10])
        print('indices ', indices[:10])
        if debug:
            raise(e)
        else:
            print(e)
            return topn, indices
    return

def render(container, **kwargs):
    resultdf = search(**kwargs)
    resultdf.style.set_properties(subset='title', **{'width':'200px'})
    resultdf.style.set_properties(subset='content', **{'width':'500px'})

    if resultdf is None:
        raise "search failed"
    else:
        container.table(resultdf.style) #, height=900, width=1200)

def main(args):
    pd.set_option('display.max_colwidth', None)
    df, embeddings = load(args.datapath)
    languages = ['any','en','es','pt'] + sorted(list(df.dropna(subset=['language']).language.unique()))

    string_factory = 'IVF256,Flat'
    print('Building index...', end='')
    index = faiss.index_factory(384, string_factory, METRIC_INNER_PRODUCT)
    index.train(embeddings)
    index.add(embeddings)
    index.nprobe = 8
    print('Done! ')
    print('Loading recommender model...',end='')
    model, head2ix = model_fn(args.modelpath)
    rep_vectors = torch.load(args.modelpath+'/rep_vectors.pt')
    print('Done!')

    c1 = st.container()
    c2 = st.container()

    du = st.sidebar.selectbox(label = 'Select your domain unit', options=sorted(list(head2ix.keys())), 
        index=0, key=None, help=None)

    lang = st.sidebar.selectbox(label = 'Select your preferred language', options=languages)

    sn = st.sidebar.text_input(label = 'Enter the serial numbers of preferred news', value='E.g. 1254,5561')

    c1.title('Recommender Demo')

    render(container = c2, **{'domain':du, 'rep_vectors':rep_vectors, 'faiss_index':index, 'df':df, 
        'head2ix':head2ix, 'embeddings':embeddings, 'model':model, 'language':lang, 'favor':sn})


if __name__ == '__main__':
    args = Args()
    main(args)
