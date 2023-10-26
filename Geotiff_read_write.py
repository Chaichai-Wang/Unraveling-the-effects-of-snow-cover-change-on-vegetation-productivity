
from osgeo import osr,gdal,gdalconst


def ReadGeoTiff(inputfile):
    ds = gdal.Open(inputfile)
    band = ds.GetRasterBand(1)
    data_arr = band.ReadAsArray()
    [Ysize, Xsize] = data_arr.shape
    return data_arr, Ysize, Xsize


def GetGeoInfo(FileName):
    SourceDS = gdal.Open(FileName)
    GeoT = SourceDS.GetGeoTransform()
    Projection = osr.SpatialReference()
    Projection.ImportFromWkt(SourceDS.GetProjectionRef())
    return GeoT, Projection


def CreateGeoTiff(Name, Array, xsize, ysize, GeoT, Projection, NoData_value):
    gdal.AllRegister()
    driver = gdal.GetDriverByName('GTiff')
    DataType = gdal.GDT_Float32
    NewFileName = Name + '.tif'
    # Set up the dataset
    DataSet = driver.Create(NewFileName, xsize, ysize, 1, DataType)
    # the '1' is for band 1.
    DataSet.SetGeoTransform(GeoT)
    DataSet.SetProjection(Projection.ExportToWkt())
    # Write the array
    DataSet.GetRasterBand(1).WriteArray(Array)

    outBand = DataSet.GetRasterBand(1)
    # flush data to disk, set the NoData value and calculate stats
    outBand.FlushCache()
    outBand.SetNoDataValue(NoData_value)
    return NewFileName


def ReprojectImages(inputfilePath,outputfilePath,referencefilefilePath):
    inputrasfile = gdal.Open(inputfilePath, gdal.GA_ReadOnly)
    inputProj = inputrasfile.GetProjection()
    referencefile = gdal.Open(referencefilefilePath, gdal.GA_ReadOnly)
    referencefileProj = referencefile.GetProjection()
    referencefileTrans = referencefile.GetGeoTransform()
    bandreferencefile = referencefile.GetRasterBand(1)
    Width= referencefile.RasterXSize
    Height = referencefile.RasterYSize
    nbands = referencefile.RasterCount

    driver = gdal.GetDriverByName('GTiff')
    output = driver.Create(outputfilePath, Width,Height, nbands, bandreferencefile.DataType)
    output.SetGeoTransform(referencefileTrans)
    output.SetProjection(referencefileProj)
    gdal.ReprojectImage(inputrasfile, output, inputProj, referencefileProj, gdalconst.GRA_NearestNeighbour)