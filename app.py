import json
import pandas as pd
import streamlit as st
import core
from core import ACTIONS, ROOT, Grant, check, keyword, resource_for, train
st.set_page_config(page_title='AWS task classification research',layout='wide')
st.title('Task classification and JIT access decisions')
st.caption('Moiz Ahmad B01841891 | Local synthetic research simulation | No AWS operations')
@st.cache_resource
def load(): return train()
data,model=load()
tab1,tab2,tab3=st.tabs(['Live demonstration','Measured results','Dataset and limitations'])
with tab1:
    text=st.text_input('Task request','Please stop the instance now')
    method=st.selectbox('Classifier',['Keyword','TFIDF_LR'])
    threshold=st.slider('Optional abstention threshold (0 = always answer; extension, not in approved scope)',0.0,0.7,0.0,0.05)
    if method=='Keyword': pred=core.keyword_or_reject(text) if threshold>0 else keyword(text)
    else: pred=core.predict_or_reject(model,[text],threshold)[0]
    st.write('Proposed action:',pred)
    if method!='Keyword': st.write('Maximum model probability (not a security guarantee):',float(model.predict_proba([text]).max()))
    allowed=st.multiselect('Trusted allowed actions',ACTIONS,default=['ec2:StopInstances'])
    target=st.text_input('Requested resource',resource_for('ec2:StopInstances'))
    trusted=st.text_input('Trusted resource',resource_for('ec2:StopInstances'))
    approved=st.checkbox('Approved',True)
    now=st.number_input('Simulated current time',value=50.0)
    start=st.number_input('Grant starts at',value=0.0)
    end=st.number_input('Grant expires at',value=100.0)
    result=check(pred,target,Grant(frozenset(allowed),frozenset([trusted]),approved,start,end),now)
    (st.success if result.allowed else st.error)(('ALLOW: ' if result.allowed else 'DENY: ')+result.reason)
    st.warning('The checker validates permission, not the meaning of the sentence. Try negation with a broad grant.')
with tab2:
    if (ROOT/'results/classification_metrics.csv').exists():
        st.dataframe(pd.read_csv(ROOT/'results/classification_metrics.csv'))
        st.dataframe(pd.read_csv(ROOT/'results/security_summary.csv'))
        for extra in ['cv_metrics','paired_difference','keyword_coverage','abstention_security']:
            if (ROOT/'results'/(extra+'.csv')).exists(): st.caption(extra); st.dataframe(pd.read_csv(ROOT/'results'/(extra+'.csv')))
        for name in ['Keyword','TFIDF_LR']: st.image(str(ROOT/'results'/(name+'_confusion.png')))
        st.dataframe(pd.read_json(ROOT/'results/challenge_results.json'))
        st.download_button('Download security cases',(ROOT/'results/security_cases.csv').read_bytes(),'security_cases.csv')
    else: st.info('Run python evaluate.py first.')
with tab3:
    st.dataframe(data)
    st.write('256 synthetic requests; 192 train and 64 held out (fixed split) plus grouped 4-fold CV over all 64 phrase families. Challenge cases are qualitative. Rules and ML both force one of eight actions. No abstention, IAM emulation, authentication or AWS execution.')
