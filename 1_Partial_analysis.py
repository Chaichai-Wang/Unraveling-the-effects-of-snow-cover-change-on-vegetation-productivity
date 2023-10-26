import sys

sys.path.append(r'E:\Code\Phenology')
import pingouin as pg
import numpy as np
import pandas as pd
from sklearn import metrics
import timeseris_interpolation as ti
import Read_Write_img as RW
import Geotiff_read_write as Geotiff_RW
import os, glob
import datetime
import matplotlib.pyplot as plt
import copy
from tqdm import tqdm

def array_growing_season(yearcount, array_factor, array_earlydate, array_latedate, mask):
    array_element = copy.deepcopy(array_factor[yearcount, :, :, :])
    array_element[(mask[yearcount, :, :, :] < array_earlydate[yearcount, :, :, :]) +
                  (mask[yearcount, :, :, :] > array_latedate[yearcount, :, :, :])] = np.nan
    array_element[array_element <= -999] = np.nan
    array_mean = np.nanmean(array_element, axis=0)
    return array_mean


def Partial_Corr(start_year, end_year, Ysize, Xsize, GeoT, Proj):
    yearnum = end_year - start_year + 1
    datenum = 365

    indir_gpp = r'E:\Result\GPP'
    indir_tmp = r'E:\Mete\Temp\Ex'
    indir_pre = r'E:\Mete\Prec\Ex'
    indir_srd = r'E:\Mete\Srad\Ex'
    # UD: upturn date
    # SD: stabilisation date
    # DD: downturn date
    # RD: recession date
    indir_upt = r'E:\GIMMS_NDVI\Phen\USDR\UD'
    indir_sta = r'E:\GIMMS_NDVI\Phen\USDR\SD'
    indir_dow = r'E:\GIMMS_NDVI\Phen\USDR\DD'
    indir_rec = r'E:\GIMMS_NDVI\Phen\USDR\RD'

    indir_sd = r'E:\Result\SWE'

    outdir = r'E:\Result\Partial_Correlation\GIMMS\EPL\EPLA\SWE'

    array_gpp = np.zeros((yearnum, datenum, Ysize, Xsize), dtype=np.float32)  # 存储原始GPP和Mete数据
    array_tmp = np.zeros((yearnum, datenum, Ysize, Xsize), dtype=np.float32)
    array_pre = np.zeros((yearnum, datenum, Ysize, Xsize), dtype=np.float32)
    array_srd = np.zeros((yearnum, datenum, Ysize, Xsize), dtype=np.float32)

    array_upt_ij_day = np.zeros((yearnum, datenum, Ysize, Xsize), dtype=np.float32)
    array_sta_ij_day = np.zeros((yearnum, datenum, Ysize, Xsize), dtype=np.float32)
    array_dow_ij_day = np.zeros((yearnum, datenum, Ysize, Xsize), dtype=np.float32)
    array_rec_ij_day = np.zeros((yearnum, datenum, Ysize, Xsize), dtype=np.float32)
    array_upt_ij = np.zeros((yearnum, Ysize, Xsize), dtype=np.float32)
    array_sta_ij = np.zeros((yearnum, Ysize, Xsize), dtype=np.float32)
    array_dow_ij = np.zeros((yearnum, Ysize, Xsize), dtype=np.float32)
    array_rec_ij = np.zeros((yearnum, Ysize, Xsize), dtype=np.float32)

    array_sd_ij = np.zeros((yearnum, Ysize, Xsize), dtype=np.float32)
    array_gpp_eg = np.zeros((yearnum, Ysize, Xsize), dtype=np.float32)  # ud-sd early growing season
    array_tmp_eg = np.zeros((yearnum, Ysize, Xsize), dtype=np.float32)
    array_pre_eg = np.zeros((yearnum, Ysize, Xsize), dtype=np.float32)
    array_srd_eg = np.zeros((yearnum, Ysize, Xsize), dtype=np.float32)
    array_gpp_pg = np.zeros((yearnum, Ysize, Xsize), dtype=np.float32)  # sd-dd peak growing season
    array_tmp_pg = np.zeros((yearnum, Ysize, Xsize), dtype=np.float32)
    array_pre_pg = np.zeros((yearnum, Ysize, Xsize), dtype=np.float32)
    array_srd_pg = np.zeros((yearnum, Ysize, Xsize), dtype=np.float32)
    array_gpp_lg = np.zeros((yearnum, Ysize, Xsize), dtype=np.float32)  # dd-rd late growing season
    array_tmp_lg = np.zeros((yearnum, Ysize, Xsize), dtype=np.float32)
    array_pre_lg = np.zeros((yearnum, Ysize, Xsize), dtype=np.float32)
    array_srd_lg = np.zeros((yearnum, Ysize, Xsize), dtype=np.float32)
    array_gpp_ag = np.zeros((yearnum, Ysize, Xsize), dtype=np.float32)  # ud-sd early growing season
    array_tmp_ag = np.zeros((yearnum, Ysize, Xsize), dtype=np.float32)
    array_pre_ag = np.zeros((yearnum, Ysize, Xsize), dtype=np.float32)
    array_srd_ag = np.zeros((yearnum, Ysize, Xsize), dtype=np.float32)
    mask = np.zeros((yearnum, datenum, Ysize, Xsize), dtype=np.float32)

    for yr in tqdm(range(start_year, end_year + 1)):
        yearcount = yr - start_year
        infile_upt = r'%s\UD%d.tif' % (indir_upt, yr + 1)
        infile_sta = r'%s\SD%d.tif' % (indir_sta, yr + 1)
        infile_dow = r'%s\DD%d.tif' % (indir_dow, yr + 1)
        infile_rec = r'%s\RD%d.tif' % (indir_rec, yr + 1)

        infile_sd = r'%s\%d.tif' % (indir_sd, yr)

        ras_upt = RW.Read_img(infile_upt)[5]
        ras_sta = RW.Read_img(infile_sta)[5]
        ras_dow = RW.Read_img(infile_dow)[5]
        ras_rec = RW.Read_img(infile_rec)[5]

        ras_sd = RW.Read_img(infile_sd)[5].astype(float)

        array_upt_ij[yearcount, :, :] = ras_upt
        array_sta_ij[yearcount, :, :] = ras_sta
        array_dow_ij[yearcount, :, :] = ras_dow
        array_rec_ij[yearcount, :, :] = ras_rec

        array_sd_ij[yearcount, :, :] = ras_sd

        ras_upt = np.reshape(np.asarray(ras_upt), [1, Ysize * Xsize]).squeeze()
        ras_sta = np.reshape(np.asarray(ras_sta), [1, Ysize * Xsize]).squeeze()
        ras_dow = np.reshape(np.asarray(ras_dow), [1, Ysize * Xsize]).squeeze()
        ras_rec = np.reshape(np.asarray(ras_rec), [1, Ysize * Xsize]).squeeze()

        ras_sd = np.reshape(np.asarray(ras_sd), [1, Ysize * Xsize]).squeeze()
        if (yr == start_year):
            array_upt = ras_upt
            array_sta = ras_sta
            array_dow = ras_dow
            array_rec = ras_rec

            array_sd = ras_sd
        else:
            array_upt = np.vstack((array_upt, ras_upt))
            array_sta = np.vstack((array_sta, ras_sta))
            array_dow = np.vstack((array_dow, ras_dow))
            array_rec = np.vstack((array_rec, ras_rec))

            array_sd = np.vstack((array_sd, ras_sd))
        array_sd[array_sd > 60000] = np.nan
        array_sd_ij[array_sd_ij > 60000] = np.nan

        for da in range(datenum):
            mask[yearcount, da, :, :] = da + 1
            day = str(da + 1).zfill(3)
            infile_gpp = r'%s\%dGPP\%d%s.tif' % (indir_gpp, yr + 1, yr + 1, day)
            infile_tmp = r'%s\%d\%d_%s_Ex.tif' % (indir_tmp, yr + 1, yr + 1, day)
            infile_pre = r'%s\%d\%d_%s_Ex.tif' % (indir_pre, yr + 1, yr + 1, day)
            infile_srd = r'%s\%d\%d_%s_Ex.tif' % (indir_srd, yr + 1, yr + 1, day)
            ras_gpp = RW.Read_img(infile_gpp)[5]
            ras_tmp = RW.Read_img(infile_tmp)[5]
            ras_pre = RW.Read_img(infile_pre)[5]
            ras_srd = RW.Read_img(infile_srd)[5]
            array_gpp[yearcount, da, :, :] = ras_gpp
            array_tmp[yearcount, da, :, :] = ras_tmp
            array_pre[yearcount, da, :, :] = ras_pre
            array_srd[yearcount, da, :, :] = ras_srd

            array_upt_ij_day[yearcount, da, :, :] = array_upt_ij[yearcount, :, :]
            array_sta_ij_day[yearcount, da, :, :] = array_sta_ij[yearcount, :, :]
            array_dow_ij_day[yearcount, da, :, :] = array_dow_ij[yearcount, :, :]
            array_rec_ij_day[yearcount, da, :, :] = array_rec_ij[yearcount, :, :]
    array_gpp[array_gpp <= -999] = np.nan
    array_tmp[array_tmp <= -999] = np.nan
    array_pre[array_pre <= -999] = np.nan
    array_srd[array_srd <= -999] = np.nan
    array_sd[array_sd <= -999] = np.nan
    array_sd_ij[array_sd_ij <= -999] = np.nan

    mean_gpp = np.nanmean(array_gpp, axis=0)
    mean_tmp = np.nanmean(array_tmp, axis=0)
    mean_pre = np.nanmean(array_pre, axis=0)
    mean_srd = np.nanmean(array_srd, axis=0)
    mean_sd = np.nanmean(array_sd, axis=0)
    mean_sd_ij = np.nanmean(array_sd_ij, axis=0)
    for yr in tqdm(range(start_year, end_year + 1)):
        yearcount = yr - start_year
        array_gpp[yearcount, :, :, :] = array_gpp[yearcount, :, :, :] - mean_gpp
        array_tmp[yearcount, :, :, :] = array_tmp[yearcount, :, :, :] - mean_tmp
        array_pre[yearcount, :, :, :] = array_pre[yearcount, :, :, :] - mean_pre
        array_srd[yearcount, :, :, :] = array_srd[yearcount, :, :, :] - mean_srd
        array_sd[yearcount, :] = array_sd[yearcount, :] - mean_sd
        array_sd_ij[yearcount, :, :] = array_sd_ij[yearcount, :, :] - mean_sd_ij

        array_gpp_eg[yearcount, :, :] = array_growing_season(yearcount, array_gpp, array_upt_ij_day, array_sta_ij_day,
                                                             mask)
        array_tmp_eg[yearcount, :, :] = array_growing_season(yearcount, array_tmp, array_upt_ij_day, array_sta_ij_day,
                                                             mask)
        array_pre_eg[yearcount, :, :] = array_growing_season(yearcount, array_pre, array_upt_ij_day, array_sta_ij_day,
                                                             mask)
        array_srd_eg[yearcount, :, :] = array_growing_season(yearcount, array_srd, array_upt_ij_day, array_sta_ij_day,
                                                             mask)
        array_gpp_pg[yearcount, :, :] = array_growing_season(yearcount, array_gpp, array_sta_ij_day, array_dow_ij_day,
                                                             mask)
        array_tmp_pg[yearcount, :, :] = array_growing_season(yearcount, array_tmp, array_sta_ij_day, array_dow_ij_day,
                                                             mask)
        array_pre_pg[yearcount, :, :] = array_growing_season(yearcount, array_pre, array_sta_ij_day, array_dow_ij_day,
                                                             mask)
        array_srd_pg[yearcount, :, :] = array_growing_season(yearcount, array_srd, array_sta_ij_day, array_dow_ij_day,
                                                             mask)
        array_gpp_lg[yearcount, :, :] = array_growing_season(yearcount, array_gpp, array_dow_ij_day, array_rec_ij_day,
                                                             mask)
        array_tmp_lg[yearcount, :, :] = array_growing_season(yearcount, array_tmp, array_dow_ij_day, array_rec_ij_day,
                                                             mask)
        array_pre_lg[yearcount, :, :] = array_growing_season(yearcount, array_pre, array_dow_ij_day, array_rec_ij_day,
                                                             mask)
        array_srd_lg[yearcount, :, :] = array_growing_season(yearcount, array_srd, array_dow_ij_day, array_rec_ij_day,
                                                             mask)
        array_gpp_ag[yearcount, :, :] = array_growing_season(yearcount, array_gpp, array_upt_ij_day, array_rec_ij_day,
                                                             mask)
        array_tmp_ag[yearcount, :, :] = array_growing_season(yearcount, array_tmp, array_upt_ij_day, array_rec_ij_day,
                                                             mask)
        array_pre_ag[yearcount, :, :] = array_growing_season(yearcount, array_pre, array_upt_ij_day, array_rec_ij_day,
                                                             mask)
        array_srd_ag[yearcount, :, :] = array_growing_season(yearcount, array_srd, array_upt_ij_day, array_rec_ij_day,
                                                             mask)

    array_gpp_eg = np.reshape(np.asarray(array_gpp_eg), [yearnum, Ysize * Xsize])
    array_tmp_eg = np.reshape(np.asarray(array_tmp_eg), [yearnum, Ysize * Xsize])
    array_pre_eg = np.reshape(np.asarray(array_pre_eg), [yearnum, Ysize * Xsize])
    array_srd_eg = np.reshape(np.asarray(array_srd_eg), [yearnum, Ysize * Xsize])
    array_gpp_pg = np.reshape(np.asarray(array_gpp_pg), [yearnum, Ysize * Xsize])
    array_tmp_pg = np.reshape(np.asarray(array_tmp_pg), [yearnum, Ysize * Xsize])
    array_pre_pg = np.reshape(np.asarray(array_pre_pg), [yearnum, Ysize * Xsize])
    array_srd_pg = np.reshape(np.asarray(array_srd_pg), [yearnum, Ysize * Xsize])
    array_gpp_lg = np.reshape(np.asarray(array_gpp_lg), [yearnum, Ysize * Xsize])
    array_tmp_lg = np.reshape(np.asarray(array_tmp_lg), [yearnum, Ysize * Xsize])
    array_pre_lg = np.reshape(np.asarray(array_pre_lg), [yearnum, Ysize * Xsize])
    array_srd_lg = np.reshape(np.asarray(array_srd_lg), [yearnum, Ysize * Xsize])
    array_gpp_ag = np.reshape(np.asarray(array_gpp_ag), [yearnum, Ysize * Xsize])
    array_tmp_ag = np.reshape(np.asarray(array_tmp_ag), [yearnum, Ysize * Xsize])
    array_pre_ag = np.reshape(np.asarray(array_pre_ag), [yearnum, Ysize * Xsize])
    array_srd_ag = np.reshape(np.asarray(array_srd_ag), [yearnum, Ysize * Xsize])

    p_corr_sd_eg = []
    p_corr_sd_pg = []
    p_corr_sd_lg = []
    p_corr_sd_ag = []
    p_sd_eg = []
    p_sd_pg = []
    p_sd_lg = []
    p_sd_ag = []

    for c in tqdm(range(Xsize * Ysize)):
        gpp_vector_eg = np.asarray(array_gpp_eg[:, c])
        tmp_vector_eg = np.asarray(array_tmp_eg[:, c])
        pre_vector_eg = np.asarray(array_pre_eg[:, c])
        srd_vector_eg = np.asarray(array_srd_eg[:, c])
        gpp_vector_pg = np.asarray(array_gpp_pg[:, c])
        tmp_vector_pg = np.asarray(array_tmp_pg[:, c])
        pre_vector_pg = np.asarray(array_pre_pg[:, c])
        srd_vector_pg = np.asarray(array_srd_pg[:, c])
        gpp_vector_lg = np.asarray(array_gpp_lg[:, c])
        tmp_vector_lg = np.asarray(array_tmp_lg[:, c])
        pre_vector_lg = np.asarray(array_pre_lg[:, c])
        srd_vector_lg = np.asarray(array_srd_lg[:, c])
        gpp_vector_ag = np.asarray(array_gpp_ag[:, c])
        tmp_vector_ag = np.asarray(array_tmp_ag[:, c])
        pre_vector_ag = np.asarray(array_pre_ag[:, c])
        srd_vector_ag = np.asarray(array_srd_ag[:, c])

        sd_vector = np.asarray(array_sd[:, c])

        ind_gpp_eg = np.isnan(gpp_vector_eg)
        ind_tmp_eg = np.isnan(tmp_vector_eg)
        ind_pre_eg = np.isnan(pre_vector_eg)
        ind_srd_eg = np.isnan(srd_vector_eg)
        ind_gpp_pg = np.isnan(gpp_vector_pg)
        ind_tmp_pg = np.isnan(tmp_vector_pg)
        ind_pre_pg = np.isnan(pre_vector_pg)
        ind_srd_pg = np.isnan(srd_vector_pg)
        ind_gpp_lg = np.isnan(gpp_vector_lg)
        ind_tmp_lg = np.isnan(tmp_vector_lg)
        ind_pre_lg = np.isnan(pre_vector_lg)
        ind_srd_lg = np.isnan(srd_vector_lg)
        ind_gpp_ag = np.isnan(gpp_vector_ag)
        ind_tmp_ag = np.isnan(tmp_vector_ag)
        ind_pre_ag = np.isnan(pre_vector_ag)
        ind_srd_ag = np.isnan(srd_vector_ag)

        ind_sd = np.isnan(sd_vector)


        ind_sd_eg = [0 for ii in range(yearnum)]
        ind_sd_pg = [0 for ii in range(yearnum)]
        ind_sd_lg = [0 for ii in range(yearnum)]
        ind_sd_ag = [0 for ii in range(yearnum)]
        for i in range(yearnum):
            ind_sd_eg[i] = (ind_gpp_eg[i] or ind_tmp_eg[i] or ind_pre_eg[i] or ind_srd_eg[i] or ind_sd[i])
            ind_sd_pg[i] = (ind_gpp_pg[i] or ind_tmp_pg[i] or ind_pre_pg[i] or ind_srd_pg[i] or ind_sd[i])
            ind_sd_lg[i] = (ind_gpp_lg[i] or ind_tmp_lg[i] or ind_pre_lg[i] or ind_srd_lg[i] or ind_sd[i])
            ind_sd_ag[i] = (ind_gpp_ag[i] or ind_tmp_ag[i] or ind_pre_ag[i] or ind_srd_ag[i] or ind_sd[i])

        ind_sd_eg = np.asarray(ind_sd_eg)
        ind_sd_pg = np.asarray(ind_sd_pg)
        ind_sd_lg = np.asarray(ind_sd_lg)
        ind_sd_ag = np.asarray(ind_sd_ag)

        gpp_sd_vector_eg_ok = gpp_vector_eg[~ind_sd_eg]
        tmp_sd_vector_eg_ok = tmp_vector_eg[~ind_sd_eg]
        pre_sd_vector_eg_ok = pre_vector_eg[~ind_sd_eg]
        srd_sd_vector_eg_ok = srd_vector_eg[~ind_sd_eg]
        sd_vector_eg_ok = sd_vector[~ind_sd_eg]

        gpp_sd_vector_pg_ok = gpp_vector_pg[~ind_sd_pg]
        tmp_sd_vector_pg_ok = tmp_vector_pg[~ind_sd_pg]
        pre_sd_vector_pg_ok = pre_vector_pg[~ind_sd_pg]
        srd_sd_vector_pg_ok = srd_vector_pg[~ind_sd_pg]
        sd_vector_pg_ok = sd_vector[~ind_sd_pg]

        gpp_sd_vector_lg_ok = gpp_vector_lg[~ind_sd_lg]
        tmp_sd_vector_lg_ok = tmp_vector_lg[~ind_sd_lg]
        pre_sd_vector_lg_ok = pre_vector_lg[~ind_sd_lg]
        srd_sd_vector_lg_ok = srd_vector_lg[~ind_sd_lg]
        sd_vector_lg_ok = sd_vector[~ind_sd_lg]

        gpp_sd_vector_ag_ok = gpp_vector_ag[~ind_sd_ag]
        tmp_sd_vector_ag_ok = tmp_vector_ag[~ind_sd_ag]
        pre_sd_vector_ag_ok = pre_vector_ag[~ind_sd_ag]
        srd_sd_vector_ag_ok = srd_vector_ag[~ind_sd_ag]
        sd_vector_ag_ok = sd_vector[~ind_sd_ag]

        data = [gpp_sd_vector_eg_ok, tmp_sd_vector_eg_ok, pre_sd_vector_eg_ok, srd_sd_vector_eg_ok, sd_vector_eg_ok,
                gpp_sd_vector_pg_ok, tmp_sd_vector_pg_ok, pre_sd_vector_pg_ok, srd_sd_vector_pg_ok, sd_vector_pg_ok,
                gpp_sd_vector_lg_ok, tmp_sd_vector_lg_ok, pre_sd_vector_lg_ok, srd_sd_vector_lg_ok, sd_vector_lg_ok,
                gpp_sd_vector_ag_ok, tmp_sd_vector_ag_ok, pre_sd_vector_ag_ok, srd_sd_vector_ag_ok, sd_vector_ag_ok]

        factors_data = ['GPP_SD_EG', 'TEMP_SD_EG', 'PREC_SD_EG', 'SRAD_SD_EG', 'SD_EG',
                        'GPP_SD_PG', 'TEMP_SD_PG', 'PREC_SD_PG', 'SRAD_SD_PG', 'SD_PG',
                        'GPP_SD_LG', 'TEMP_SD_LG', 'PREC_SD_LG', 'SRAD_SD_LG', 'SD_LG',
                        'GPP_SD_AG', 'TEMP_SD_AG', 'PREC_SD_AG', 'SRAD_SD_AG', 'SD_AG']

        df = pd.DataFrame(data, index=factors_data).T


        if (len(sd_vector_eg_ok) >= 10):
            try:
                pcorr_sd_eg = pg.partial_corr(data=df, x='SD_EG', y='GPP_SD_EG',
                                              covar=['TEMP_SD_EG', 'PREC_SD_EG', 'SRAD_SD_EG'])
                p_corr_sd_eg.append(pcorr_sd_eg['r'].values)
                p_sd_eg.append(pcorr_sd_eg['p-val'].values)
            except:
                p_corr_sd_eg.append(np.nan)
                p_sd_eg.append(np.nan)
        else:
            p_corr_sd_eg.append(np.nan)
            p_sd_eg.append(np.nan)


        if (len(sd_vector_pg_ok) >= 10):
            try:
                pcorr_sd_pg = pg.partial_corr(data=df, x='SD_PG', y='GPP_SD_PG',
                                              covar=['TEMP_SD_PG', 'PREC_SD_PG', 'SRAD_SD_PG'])
                p_corr_sd_pg.append(pcorr_sd_pg['r'].values)
                p_sd_pg.append(pcorr_sd_pg['p-val'].values)
            except:
                p_corr_sd_pg.append(np.nan)
                p_sd_pg.append(np.nan)
        else:
            p_corr_sd_pg.append(np.nan)
            p_sd_pg.append(np.nan)


        if (len(sd_vector_lg_ok) >= 10):
            try:
                pcorr_sd_lg = pg.partial_corr(data=df, x='SD_LG', y='GPP_SD_LG',
                                              covar=['TEMP_SD_LG', 'PREC_SD_LG', 'SRAD_SD_LG'])
                p_corr_sd_lg.append(pcorr_sd_lg['r'].values)
                p_sd_lg.append(pcorr_sd_lg['p-val'].values)
            except:
                p_corr_sd_lg.append(np.nan)
                p_sd_lg.append(np.nan)
        else:
            p_corr_sd_lg.append(np.nan)
            p_sd_lg.append(np.nan)


        if (len(sd_vector_ag_ok) >= 10):
            try:
                pcorr_sd_ag = pg.partial_corr(data=df, x='SD_AG', y='GPP_SD_AG',
                                              covar=['TEMP_SD_AG', 'PREC_SD_AG', 'SRAD_SD_AG'])
                p_corr_sd_ag.append(pcorr_sd_ag['r'].values)
                p_sd_ag.append(pcorr_sd_ag['p-val'].values)
            except:
                p_corr_sd_ag.append(np.nan)
                p_sd_ag.append(np.nan)
        else:
            p_corr_sd_ag.append(np.nan)
            p_sd_ag.append(np.nan)


    p_corr_sd_eg = np.reshape(p_corr_sd_eg, [Ysize, Xsize])
    p_sd_eg = np.reshape(p_sd_eg, [Ysize, Xsize])

    p_corr_sd_pg = np.reshape(p_corr_sd_pg, [Ysize, Xsize])
    p_sd_pg = np.reshape(p_sd_pg, [Ysize, Xsize])

    p_corr_sd_lg = np.reshape(p_corr_sd_lg, [Ysize, Xsize])
    p_sd_lg = np.reshape(p_sd_lg, [Ysize, Xsize])

    p_corr_sd_ag = np.reshape(p_corr_sd_ag, [Ysize, Xsize])
    p_sd_ag = np.reshape(p_sd_ag, [Ysize, Xsize])


    outfile_rel_sd_eg = '%s\GPP_SWE_EG_pcorr' % (outdir)
    Geotiff_RW.CreateGeoTiff(outfile_rel_sd_eg, p_corr_sd_eg, Xsize, Ysize, GeoT, Proj, np.nan)
    outfile_p_sd_eg = '%s\GPP_SWE_EG_p' % (outdir)
    Geotiff_RW.CreateGeoTiff(outfile_p_sd_eg, p_sd_eg, Xsize, Ysize, GeoT, Proj, np.nan)


    outfile_rel_sd_pg = '%s\GPP_SWE_PG_pcorr' % (outdir)
    Geotiff_RW.CreateGeoTiff(outfile_rel_sd_pg, p_corr_sd_pg, Xsize, Ysize, GeoT, Proj, np.nan)
    outfile_p_sd_pg = '%s\GPP_SWE_PG_p' % (outdir)
    Geotiff_RW.CreateGeoTiff(outfile_p_sd_pg, p_sd_pg, Xsize, Ysize, GeoT, Proj, np.nan)


    outfile_rel_sd_lg = '%s\GPP_SWE_LG_pcorr' % (outdir)
    Geotiff_RW.CreateGeoTiff(outfile_rel_sd_lg, p_corr_sd_lg, Xsize, Ysize, GeoT, Proj, np.nan)
    outfile_p_sd_lg = '%s\GPP_SWE_LG_p' % (outdir)
    Geotiff_RW.CreateGeoTiff(outfile_p_sd_lg, p_sd_lg, Xsize, Ysize, GeoT, Proj, np.nan)


    outfile_rel_sd_ag = '%s\GPP_SWE_AG_pcorr' % (outdir)
    Geotiff_RW.CreateGeoTiff(outfile_rel_sd_ag, p_corr_sd_ag, Xsize, Ysize, GeoT, Proj, np.nan)
    outfile_p_sd_ag = '%s\GPP_SWE_AG_p' % (outdir)
    Geotiff_RW.CreateGeoTiff(outfile_p_sd_ag, p_sd_ag, Xsize, Ysize, GeoT, Proj, np.nan)


if __name__ == "__main__":
    start_time = datetime.datetime.now()
    refer_tif_file = r'E:/Project_para/1981_ex.tif'
    array, Ysize, Xsize = Geotiff_RW.ReadGeoTiff(refer_tif_file)
    [GeoT, Proj] = Geotiff_RW.GetGeoInfo(refer_tif_file)
    Partial_Corr(1981, 2014, Ysize, Xsize, GeoT, Proj)
    end_time = datetime.datetime.now()
    print('time：' + str(end_time - start_time))
