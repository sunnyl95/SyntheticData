from syntheticdata import demo

tables = demo.load_tabular_demo()

tables = tables.head(1000)

# from sdv.tabular.ctgan import CTGAN
#from  syntheticdata.tabular.model_ctgan import  CTGAN
# model = CTGAN(epochs=10)
# model.fit(tables)


from  syntheticdata.tabular.model_tvae import  TVAE
model = TVAE(epochs=10)
model.fit(tables)

