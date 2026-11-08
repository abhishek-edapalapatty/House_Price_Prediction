from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data'/'house_price.csv'
MODEL=ROOT/'models'/'house_price_pipeline.joblib'
TARGET='TARGET(PRICE_IN_LACS)'
FEATURES=['POSTED_BY','UNDER_CONSTRUCTION','RERA','BHK_NO.','BHK_OR_RK','SQUARE_FT','READY_TO_MOVE','RESALE','ADDRESS','LONGITUDE','LATITUDE']
st.set_page_config(page_title='House Price Estimator',page_icon='🏠',layout='wide')

@st.cache_data
def load_data():
    return pd.read_csv(DATA)
@st.cache_resource
def load_model():
    return joblib.load(MODEL)

def main():
    st.title('House Price Estimator')
    st.caption('Estimate listing prices in lakhs using the supplied India property dataset. This is an educational estimate, not a certified valuation.')
    if not MODEL.exists():
        st.warning('Model artifact is missing. In the project terminal run: python src/train_model.py')
        st.stop()
    tab_est,tab_data,tab_about=st.tabs(['Estimate a property','Explore the dataset','About the model'])
    df=load_data()
    with tab_est:
        st.subheader('Property details')
        cities=sorted({str(x).split(',')[-1].strip().title() for x in df['ADDRESS'].dropna()})
        addresses=sorted(df['ADDRESS'].dropna().astype(str).unique())
        a,b,c=st.columns(3)
        with a:
            posted=st.selectbox('Listing posted by',sorted(df.POSTED_BY.unique()))
            status=st.selectbox('Property type',sorted(df['BHK_OR_RK'].unique()))
            bhk=st.slider('BHK count',1,20,2)
            area=st.number_input('Area (square feet)',min_value=3.0,max_value=100000.0,value=1200.0,step=50.0)
        with b:
            under=st.selectbox('Under construction',[0,1],format_func=lambda x:'Yes' if x else 'No')
            rera=st.selectbox('RERA approved',[0,1],format_func=lambda x:'Yes' if x else 'No')
            ready=st.selectbox('Ready to move',[1,0],format_func=lambda x:'Yes' if x else 'No')
            resale=st.selectbox('Resale listing',[1,0],format_func=lambda x:'Yes' if x else 'No')
        with c:
            addr=st.selectbox('Known address',addresses,index=0)
            sample=df.loc[df.ADDRESS.eq(addr)].iloc[0]
            longitude=st.number_input('Longitude',value=float(sample.LONGITUDE),format='%.6f')
            latitude=st.number_input('Latitude',value=float(sample.LATITUDE),format='%.6f')
        if st.button('Estimate price',type='primary'):
            row=pd.DataFrame([{'POSTED_BY':posted,'UNDER_CONSTRUCTION':under,'RERA':rera,'BHK_NO.':bhk,'BHK_OR_RK':status,'SQUARE_FT':area,'READY_TO_MOVE':ready,'RESALE':resale,'ADDRESS':addr,'LONGITUDE':longitude,'LATITUDE':latitude}])[FEATURES]
            prediction=float(load_model().predict(row)[0])
            st.metric('Estimated listing price',f'₹ {prediction:,.2f} lakhs')
            st.caption('Model estimates one point value. It does not provide a confidence interval or account for market changes after the training data was collected.')
    with tab_data:
        st.subheader('Dataset overview')
        st.write(f'{len(df):,} records · {df.shape[1]} columns · {df.isna().sum().sum():,} missing cells')
        x,y,z=st.columns(3)
        x.metric('Median listed price',f"₹ {df[TARGET].median():,.1f} L")
        y.metric('Median area',f"{df.SQUARE_FT.median():,.0f} sq ft")
        z.metric('Locations',f"{df.ADDRESS.nunique():,}")
        st.caption('Interactive plots are restricted to ordinary display quantiles so extreme records do not flatten the charts. Source rows remain unchanged.')
        q=float(df[TARGET].quantile(.99))
        st.bar_chart(df.loc[df[TARGET]<=q].groupby('BHK_NO.')[TARGET].median().rename('Median price (lakhs)'))
        st.bar_chart(df['POSTED_BY'].value_counts())
        with st.expander('Show source rows'):
            st.dataframe(df, use_container_width=True, height=360)
        st.download_button('Download source CSV',DATA.read_bytes(),'house_price.csv','text/csv')
    with tab_about:
        st.subheader('How the estimate is produced')
        st.markdown('The fitted scikit-learn pipeline imputes missing values, one-hot encodes listing agent, property type, and address, scales numeric inputs, and predicts log-transformed price. See `reports/Model_Development_Document.docx` for data review, model selection, limitations, and validation details.')
        st.markdown('**Supported interpretation:** an estimate for listings similar to the supplied historical sample. **Not supported:** guaranteed sale price, investment advice, mortgage appraisal, or live market quote.')
if __name__=='__main__': main()
