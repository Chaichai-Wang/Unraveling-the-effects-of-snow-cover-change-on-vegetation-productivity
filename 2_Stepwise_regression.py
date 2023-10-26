# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import statsmodels.api as sm
import os, glob, sys
import Read_Write_img as RW
from sklearn import preprocessing
from osgeo import gdal
from tqdm import tqdm
sys.path.append(r'E:\Code\Phenology')

def ReadRasterProp(ds):
    Prop = {}
    geotransform = ds.GetGeoTransform()
    Prop["bandCount"] = ds.RasterCount
    Prop["xcount"] = ds.RasterXSize
    Prop["ycount"] = ds.RasterYSize
    Prop["geotransform"] = ds.GetGeoTransform()
    Prop["xsize"] = geotransform[1]
    Prop["x"] = geotransform[2]
    Prop["y"] = geotransform[4]
    Prop["ysize"] = abs(geotransform[5])
    Prop["dataType"] = ds.GetRasterBand(1).DataType
    Prop["proj_info"] = ds.GetProjection()
    Prop["geotrans"] = ds.GetGeoTransform()
    Prop["xlt"] = geotransform[0]
    Prop["ylt"] = geotransform[3]
    return Prop


def stepwise(X, y, alpha_in, alpha_out):

    included = []  # list of features to start with (column names of X)
    while True:
        changed = False
        excluded = list(set(X.columns) - set(included))
        p_val = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(X[included + [new_column]])).fit()
            p_val[new_column] = model.pvalues[new_column]
        min_pval = p_val.min()
        # forward step
        if min_pval < alpha_in:
            changed = True
            add_feature = p_val.idxmin()
            included.append(add_feature)
            print("Add {:20} with p_value   {:.6}".format(add_feature, min_pval))
        # backward step
        model = sm.OLS(y, sm.add_constant(X[included])).fit()
        p_val = model.pvalues.iloc[1:]
        max_pval = p_val.max()  # null if pvalues is empty
        if max_pval > alpha_out:
            changed = True
            drop_feature = p_val.idxmax()
            included.remove(drop_feature)
            print("Drop {:20} with p_value   {:.6}".format(drop_feature, max_pval))
        if not changed:
            break
    return included


def Read_array(InPath):
    image_list_data = glob.glob(os.path.join(InPath, "*.tif"))
    scd_array = []
    for img in image_list_data:
        array_scd = RW.Read_img(img)[5]
        scd_array.append(array_scd)
    scd_array = np.array(scd_array)
    return scd_array

def Cal_contribution(x1, x2, x3, y, start_year, end_year, OutPath):
    RawData = RW.Read_img(r'E:/Project_para/1981_ex.tif')
    Img_width, Img_height, Img_proj, Img_Geo = RawData[0], RawData[1], RawData[3], RawData[4]
    name1, name2, name3, name5 = 'SCD', 'SWE', 'SCM', 'GPP'
    data1 = np.zeros(x1[0].shape, dtype=np.float32)
    data2 = np.zeros(x1[0].shape, dtype=np.float32)
    data3 = np.zeros(x1[0].shape, dtype=np.float32)
    data4 = np.zeros(x1[0].shape, dtype=np.float32)
    data5 = np.zeros(x1[0].shape, dtype=np.float32)
    data6 = np.zeros(x1[0].shape, dtype=np.float32)

    for i in range(0, x1[0].shape[0]):
        for j in range(0, x1[0].shape[1]):
            list1, list2, list3, list5 = [], [], [], []
            for t in range(end_year - start_year + 1):
                if x1[t, i, j] < 10000 and x2[t, i, j] < 10000 and x3[t, i, j] < 10000 and y[t, i, j] < 10000:
                    list1.append(x1[t, i, j]), list2.append(x2[t, i, j]), list3.append(x3[t, i, j]), \
                    list5.append(y[t, i, j])
            if len(list1) > 20:
                df = pd.DataFrame(data=[list1, list2, list3, list5], index=[name1, name2, name3, name5]).T
                maxmin = preprocessing.MinMaxScaler()
                df = maxmin.fit_transform(df)
                factor = [name1, name2, name3]
                df_zscore = pd.DataFrame(df, columns=[name1, name2, name3, name5])
                x, y1 = df_zscore.iloc[:, 0:3], list(df_zscore.iloc[:, 3])
                result = stepwise(x, y1, alpha_in=0.1, alpha_out=0.1)
                print(result)
                model = sm.OLS(y1, sm.add_constant(x[result])).fit()
                Para = model.params
                R2 = model.rsquared
                PValue = model.pvalues
                constant = Para[0]
                for inde in range(1, Para.shape[0]):
                    sel_factor = Para.index.values[inde]
                    factor.remove(sel_factor)
                    if sel_factor == 'SCD':
                        SCD = Para.loc[sel_factor]
                    elif sel_factor == 'SWE':
                        SWE = Para.loc[sel_factor]
                    elif sel_factor == 'SCM':
                        PREC = Para.loc[sel_factor]
                for f in factor:
                    if f == 'SCD':
                        SCD = np.nan
                    elif f == 'SWE':
                        SWE = np.nan
                    elif f == 'SCM':
                        PREC = np.nan
            else:
                SCD, SWE, PREC, R2, constant, PValue = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
            data1[i, j], data2[i, j], data3[i, j], data5[i, j], data6[i, j] = SCD, SWE, PREC, R2, constant

    # return out_corr
    outfile_SCD = OutPath + '/' + 'stepwise' + '_SCD1_' + str(start_year) + '-' + str(end_year) + '.tif'
    outfile_SWE = OutPath + '/' + 'stepwise' + '_SWE1_' + str(start_year) + '-' + str(end_year) + '.tif'
    outfile_PREC = OutPath + '/' + 'stepwise' + '_SCM1_' + str(start_year) + '-' + str(end_year) + '.tif'
    outfile_R2 = OutPath + '/' + 'stepwise' + '_R21_' + str(start_year) + '-' + str(end_year) + '.tif'
    outfile_constant = OutPath + '/' + 'stepwise' + '_constant1_' + str(start_year) + '-' + str(end_year) + '.tif'
    # 数组输出图像
    RW.Write_img(outfile_SCD, Img_proj, Img_Geo, data1)
    RW.Write_img(outfile_SWE, Img_proj, Img_Geo, data2)
    RW.Write_img(outfile_PREC, Img_proj, Img_Geo, data3)
    RW.Write_img(outfile_R2, Img_proj, Img_Geo, data5)
    RW.Write_img(outfile_constant, Img_proj, Img_Geo, data6)

phen = "EG"
indir_gpp = r'E:\Result\Factor_EPLA\GPP\%s' % (phen)
indir_sd = r'E:\Result\Anomaly\SWE_Jiang'
indir_scd = r'E:\Result\Anomaly\SCD'
indir_sced = r'E:\Result\Anomaly\SCED'
OutPath = r'E:\Result\StepWise\Jiang_GPP_new\%s' % (phen)
start_year = 1981
end_year = 2014
array_sced = []
array_scd = []
array_sd = []
array_gpp = []
for yr in tqdm(range(start_year, end_year + 1)):
    yearcount = yr - start_year
    infile_sced = r'%s\%d.tif' % (indir_sced, yr)
    infile_scd = r'%s\%d.tif' % (indir_scd, yr)
    infile_sd = r'%s\%d.tif' % (indir_sd, yr)
    infile_gpp = r'%s\%d.tif' % (indir_gpp, yr + 1)
    ras_sced = RW.Read_img(infile_sced)[5].astype(float)
    ras_scd = RW.Read_img(infile_scd)[5].astype(float)
    ras_sd = RW.Read_img(infile_sd)[5].astype(float)
    ras_gpp = RW.Read_img(infile_gpp)[5]
    array_sced.append(ras_sced)
    array_scd.append(ras_scd)
    array_sd.append(ras_sd)
    array_gpp.append(ras_gpp)

array_sced = np.array(array_sced)
array_scd = np.array(array_scd)
array_sd = np.array(array_sd)
array_gpp = np.array(array_gpp)

Cal_contribution(array_scd, array_sd, array_sced, array_gpp, start_year, end_year, OutPath)
