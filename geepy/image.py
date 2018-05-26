"""
General Image functions
"""

import ee
import errno
import pandas as pd

from geepy.feature import setGeometry

from socket import error as SocketError

ee.Initialize()

def getInfo(obj):
    """
    getInfo docstring
    
    Args:
        obj: 
    """
    try:
        return [obj[0], obj[1].divide(obj[2]).getInfo()]
    except SocketError as e:
        if e.errno != errno.ECONNRESET:
            raise

        print(e.errno)
        pass


def edgeRemoval(img, bufferSize = -9000):
    """
    Removes the edges of an img.

    Args:
        img: The input img.
        bufferSize: The number of pixels to remove. Negative number indicates an inverse buffer.
    """
    bbox = img.geometry()
    
    return img.clip(bbox.buffer(bufferSize).simplify(1))


def clipImageFromFusionTable(img, tile):
    """
    clipImageFromFusionTable docstring
    
    Args:
        img: Image to crop.
        tile: Region to crop img.
    """
    return img.clip(tile.geometry())


def imgMask(img, pixValue):
    """
    imgMask docstring
    
    Args:
        img: Image to mask.
        pixValue: Pixel value to mask Image.
    """
    mask = img.eq(pixValue)
    return mask


def waterMask(img, bandName = 'wmask'):
    """
    Get the wmask band from an img with MNDWI band.

    Args:
        img: Image containing the MNDWI band.
    """
    wmask = img.select(['mndwi']).gt(0)
    
    return (img.addBands(wmask.rename([bandName])))


def send2drive(img, coords, desc, driveFolder, scale = 30):
    """
    Creates a batch task to export an Image to Google Drive.
    
    Args:
        img: The img to export to Google Drive.
        coords: The bounding box of the img.
        desc: A description to show in Earth Engine Code Editor.
        driveFolder: The Google Drive Folder that the export will reside in.
        scale: Resolution in meters per pixel.
    """
    task2Drive = ee.batch.Export.image.toDrive(
        image = img,
        folder = driveFolder,
        description = desc,
        maxPixels = 1e10,
        region = coords,
        scale = scale,
    )
    return task2Drive


def send2asset(img, coords, desc, assetId, scale = 30):
    """
    Creates a batch task to export an Image to an Earth Engine asset.
    
    Args:
        img: Image to save in Asset.
        coords: The bounding box of the img.
        desc: The img description.
        assetID: The destination asset ID.
        scale: Resolution in meters per pixel.
    """
    task2asset = ee.batch.Export.image.toAsset(
        image = img,
        assetId = assetId,
        description = desc,
        maxPixels = 1e10,
        region = coords,
        scale = scale
    )
    return task2asset


def tassCapTransformation(img, satellite):
    """
        Performs Tasseled Cap (Kauth Thomas) transformation.
        
        Args:
            img: The input landsat img with blue, green, red, nir, swir1 and swir2 bands.
            satellite: The satellite source l5: Landsat 5, l7: Landsat 7, l8: Landsat 8 OLI.
            
        References:
        L5 coefficients
        ERIC P. CRIST AND RICHARD C. CICONE.,
        A Physically-Based Transformation of Thematic Mapper Data-The TM Tasseled Cap. (1984)
        IEEE TRANSACTIONS ON GEOSCIENCE AND REMOTE SENSING,
        
        L7 coefficients
        Huang, C.; Yang, Wylie L.; Homer, Collin; and Zylstra, G., 
        Derivation of a tasselled cap transformation based on Landsat 7 at-satellite reflectance. (2002). 
        USGS Staff -- Published Research. 621. http://digitalcommons.unl.edu/usgsstaffpub/621

        L8 coefficients
        Muhammad Hasan Ali Baig, Lifu Zhang, Tong Shuai & Qingxi Tong (2014)
        Derivation of a tasselled cap transformation based on Landsat 8 at-satellite reflectance,
        Remote Sensing Letters, 5:5, 423-431, DOI: 10.1080/2150704X.2014.915434

    """
    columns = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']
    index = ['Brightness', 'Greenness', 'Wetness', 'TC4', 'TC5', 'TC6']
    
    bands = img.select(columns)
    
    coeffs = {'l5':[[0.3037,0.2793,0.4743,0.5585,0.5082,0.1863],
              [-0.2848,-0.2435,-0.5436,0.7243,0.0840,-0.1800],
              [0.1509,0.1973,0.3279,0.3406,-0.7112,-0.4572],
              [-0.8242,0.0849,0.4392,-0.0580,0.2012,-0.2768],
              [-0.3280,0.0549,0.1075,0.1855,-0.4357,0.8085],
              [0.1084,-0.9022,0.4120,0.0573,-0.0251,0.0238]],
              'l7':[[0.3561,0.3972,0.3904,0.6966,0.2286,0.1596],
              [-0.3344,-0.3544,-0.4556,0.6966,-0.0242,-0.2630],
              [0.2626,0.2141,0.0926,0.0656,-0.7629,-0.5388],
              [0.0805,-0.0498,0.1950,-0.1327,0.5752,-0.7775],
              [-0.7252,-0.0202,0.6683,0.0631,-0.1494,-0.0274],
              [0.4000,-0.8172,0.3832,0.0602,-0.1095,0.0985]],
              'l8':[[0.3029,0.2786,0.4733,0.5599,0.5080,0.1872],
              [-0.2941,-0.2430,-0.5424,0.7276,0.0713,-0.1608],
              [0.1511,0.1973,0.3283,0.3407,-0.7117,-0.4559],
              [-0.8239,0.0849,0.4396,-0.0580,0.2013,-0.2773],
              [-0.3294,0.0557,0.1056,0.1855,-0.4349,0.8085],
              [0.1079,-0.9023,0.4119,0.0575,-0.0259,0.0252]]}

    coeffs = pd.DataFrame(coeffs[satellite], index = index, columns = columns)

    brightness = img.expression('(k * brightness)', {
            'k':bands,
            'brightness': ee.Image(list(coeffs.loc['Brightness']))
        })
    
    greenness = img.expression('(k * greenness)', {
            'k':bands,
            'greenness': ee.Image(list(coeffs.loc['Greenness']))
        })
    
    wetness = img.expression('(k * wetness)', {
            'k':bands,
            'wetness': ee.Image(list(coeffs.loc['Wetness']))
        })
    
    TC4 = img.expression('(k * TC4)', {
            'k':bands,
            'TC4': ee.Image(list(coeffs.loc['TC4']))
        }) 

    TC5 = img.expression('(k * TC5)', {
            'k':bands,
            'TC5': ee.Image(list(coeffs.loc['TC5']))
        }) 

    TC6 = img.expression('(k * TC6)', {
            'k':bands,
            'TC6': ee.Image(list(coeffs.loc['TC6']))
        }) 

    brightness = brightness.reduce(ee.call('Reducer.sum')).rename('brightness');
    greenness = greenness.reduce(ee.call('Reducer.sum')).rename('greenness');
    wetness = wetness.reduce(ee.call('Reducer.sum')).rename('wetness');
    TC4 = TC4.reduce(ee.call('Reducer.sum')).rename('TC4');
    TC5 = TC5.reduce(ee.call('Reducer.sum')).rename('TC5');
    TC6 = TC6.reduce(ee.call('Reducer.sum')).rename('TC6');
    
    tassCap = ee.Image(brightness).addBands(greenness).addBands(wetness).addBands(TC4).addBands(TC5).addBands(TC6)
    
    return img.addBands(tassCap)


def calcBCI(img, geometry, bandName = 'bci'):
    """ 
    Calculates the Biophysical Composition Index (BCI).
    
    Args:
        img: The input img with Tasseled Cap bands.
        bandName: The output band name.
    """

    b = img.select('brightness').reduceRegion(
        reducer = ee.Reducer.minMax(),
        geometry = geometry,
        scale = 30,
        maxPixels = 1e13
    )

    g = img.select('greenness').reduceRegion(
        reducer = ee.Reducer.minMax(),
        geometry = geometry,
        scale = 30,
        maxPixels = 1e13
    )

    w = img.select('wetness').reduceRegion(
        reducer = ee.Reducer.minMax(),
        geometry = geometry,
        scale = 30,
        maxPixels = 1e13
    )

    H = img.expression('(brightness - min)/(max - min)',{
        'brightness': img.select('brightness'),
        'min': b.getInfo()['brightness_min'],
        'max': b.getInfo()['brightness_max']
    }).rename('H')

    V = img.expression('(greenness - min)/(max - min)',{
        'greenness': img.select('greenness'),
        'min': g.getInfo()['greenness_min'],
        'max': g.getInfo()['greenness_max']
    }) .rename('V')

    L = img.expression('(wetness - min)/(max - min)',{
        'wetness': img.select('wetness'),
        'min': w.getInfo()['wetness_min'],
        'max': w.getInfo()['wetness_max']
    }).rename('L')

    HVL = ee.Image(H).addBands(V).addBands(L)
    
    BCI = HVL.expression('(0.5 * (H + L) - V)/(0.5* (H + L) + V)', {
        'H': HVL.select('H'),
        'V': HVL.select('V'),
        'L': HVL.select('L')
    })
    
    return (img.addBands(
        HVL.expression('(0.5 * (H + L) - V)/(0.5* (H + L) + V)', {
            'H': HVL.select('H'),
            'V': HVL.select('V'),
            'L': HVL.select('L')
        }).rename([bandName])))


def calcEVI(img, bandName = 'evi'):
    """
    Calculates the Enhanced Vegetation Index (EVI).

    Args:
        img: The input image.
        bandName: The output band name.
    """
    return (img.addBands(
        img.expression('2.5 * float(nir - red) / (nir + 6 * red - 7.5 * blue + 1)', {
            'nir':img.select('nir'),
            'red':img.select('red'),
            'blue': img.select('blue')
        }).rename([bandName])))


def calcNDWI(img, bandName = 'ndwi'):
    """
    Calculates the Normalized Difference Water Index (NDWI).

    Args:
        img: THe input image.
        bandName: The output band name.
    """
    return (img.addBands(
        img.expression('float(nir - swir1)/(nir + swir1)', {
            'nir':img.select('nir'),
            'swir1':img.select('swir1')
        }).rename([bandName])))


def calcMNDWI(img, bandName = 'mndwi'):
    """
    Calculates the Modified Normalized Difference Water Index (MNDWI).

    Args:
        img: The input image.
        bandName: The output band name.
    """
    return (img.addBands(
        img.expression('float(green - nir)/(green + nir)', {
            'green':img.select('green'),
            'nir':img.select('nir')
        }).rename([bandName])))


def calcNDVI(img, bandName = 'ndvi'):
    """
    Calculates the Normalized Difference Vegetation Index (NDVI).

    Args:
        img: The input image.
        bandName: The output band name.
    """
    return (img.addBands(
        img.expression('float(nir - red)/(nir + red)', {
            'nir':img.select('nir'),
            'red':img.select('red')
        }).rename([bandName])))


def calcSAVI(img, bandName = 'savi'):
    """
    Calculates the Soil Adjusted Vegetation Index (SAVI).

    Args:
        img: The input image.
        bandName: The output band name.
    """
    return (img.addBands(
        img.expression('(1 + L) * float(nir - red)/(nir + red + L)', {
            'nir':img.select('nir'),
            'red':img.select('red'),
            'L': 0.9
        }).rename([bandName])))


def calcNDFI(img, bandName = 'ndfi'):
    """
    Calculates the Normalized Difference Fraction Index (NDFI).

    Args:
        img: The input image.
        bandNames: The output band name.
    """
    gvs = img.select("gvs")
    
    npvSoil = img.select("npv").add(img.select("soil"));
    
    ndfi = ee.Image.cat(gvs, npvSoil).normalizedDifference()
    
    ndfi = ndfi.multiply(100).add(100)
    
    return img.addBands(ndfi.rename(bandName))


def calcNDFI2(img, bandName = 'ndfi2'):
    """
    Calculates the Normalized Difference Fraction Index (NDFI).

    Args:
        img: The input image.
        bandNames: The output band name.
    """
    gvnpv = calcGVNPV(img).select('gvnpv')

    soilcloud = img.expression('soil + cloud', {
        'soil': img.select('soil'),
        'cloud': img.select('cloud'),
    })

    ndfi2 = ee.Image.cat(gvnpv, soilcloud).normalizedDifference().multiply(100).add(100)
   
    return img.addBands(ndfi2.rename([bandName]))


def calcNDFI3(img, bandName = 'ndfi3'):
    """
    Calculates the Normalized Difference Fraction Index (NDFI).

    Args:
        img: The input image.
        bandNames: The output band name.
    """
    gvnpv = img.expression('gv + npv', {
        'gv': img.select('gv'),
        'npv': img.select('npv'),
    })

    soilshade = img.expression('soil + shade', {
        'soil': img.select('soil'),
        'shade': img.select('shade'),
    })

    ndfi3 = ee.Image.cat(gvnpv, soilshade).normalizedDifference().multiply(100).add(100)

    return img.addBands(ndfi3.rename([bandName]))


def calcFCI(img, bandName = 'fci'):
    """
    Calculates the Forest and Crop Index (FCI).

    Args:
        img: The input image.
        bandNames: The output band name.
    """
    fci = img.expression('(float(gv - shade)/(gv + shade))', {
        'gv': img.select('gv'),
        'shade': img.select('shade'),
    }).multiply(100).add(100)
    
    return img.addBands(fci.rename([bandName]))


def calcGVNPV(img, bandName = 'gvnpv'):
    gvnpv = img.expression('100 * float(gv + npv) / float(100 - shade)', {
        'gv': img.select('gv'),
        'npv': img.select('npv'),
        'shade': img.select('shade'),
    })

    return img.addBands(gvnpv.rename([bandName]))


def calcNPVSOIL(img, bandName = 'npvsoil'):
    npvsoil = img.expression('npv + soil', {
        'npv': img.select('npv'),
        'soil': img.select('soil'),
    })

    return img.addBands(npvsoil.rename([bandName]))


def calcNDBI(img, bandName = 'ndbi'):
    """
    Calculates the Normalized Difference Built Index (NDBI).

    Args:
        img: The input image.
        bandNames: The output band name.
    """
    ndbi = img.expression('(swir1 - nir)/(swir1 + nir)', {
        'swir1': img.select('swir1'),
        'nir': img.select('nir'),
    })

    return img.addBands(ndbi.rename([bandName]))


def calcFractions(img):
    """
        calcFractions help
    """
    endmembers = [
            [119.0, 475.0, 169.0, 6250.0, 2399.0, 675.0],     # gv
            [1514.0, 1597.0, 1421.0, 3053.0, 7707.0, 1975.0], #npv
            [1799.0, 2479.0, 3158.0, 5437.0, 7707.0, 6646.0], #soil
            [4031.0, 8714.0, 7900.0, 8989.0, 7002.0, 6607.0]  #cloud
    ]
    
    bandNames = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']
    outBandNames = ['gv', 'npv', 'soil', 'cloud']

    #Unmixing data
    fractions = ee.Image(img).select(bandNames).unmix(endmembers).max(0).multiply(10000)
    
    fractions = fractions.select([0, 1, 2, 3], outBandNames)
    
    summed = fractions.select(['gv', 'npv', 'soil']).reduce(ee.Reducer.sum())
    
    shd = summed.abs()
    
    gvs = fractions.select(["gv"]).divide(summed).multiply(100)
        
    gvs = gvs.rename("gvs")
    shd = shd.rename("shade")
    
    return img.addBands(fractions).addBands(gvs).addBands(shd)


def img2Band(img, band, bandName):
    """
        img2Band help
    """
    
    return img.addBands(band.rename([str(bandName)]))


def randomSamples(feature, img, numPoints, seed, classBand, classValues, classPoints, scale = 30):
    """
    Samples the pixels of an image in one or more regions, returning them as a FeatureCollection.
    Each output feature will have 1 property per band in the input image, as well as any specified 
    properties copied from the input feature.
        
        Args:
            feature: A feature collection with region.
            img: The input image with classes to sample.
            numPoints: The number of samples.
            classBand: The attribute to sample.
            classValues: A list of values to sample.
            classPoints: A list with number of sample in each classValue.
            scale: Resolution in meters per pixel.
    """
    img = img.addBands(ee.Image.pixelLonLat())
    
    points = img.stratifiedSample(
        numPoints = numPoints,
        classBand = classBand, 
        region = feature,
        seed = seed,
        classValues = classValues, # valores a serem classificados 
        classPoints = classPoints,     
        dropNulls = True, 
        scale = scale
    )
    
    points = points.randomColumn('randCol', 0)

    return points.map(setGeometry)


def trainingSamples(img, samples, classBand = 'CLASS', scale = 30):
    """
        training help

        Args:
            img:
            samples:
            classBand:
            scale: Resolution in meters per pixel.
    """
    training = img.sampleRegions(
        collection = samples,
        properties = [classBand],
        scale = scale,
    )

    return training


def randomForest(img, training, bands, ntrees = 10, classBand = 'CLASS', scale = 10):
    """
        randomForest help
        
        Args:
            img:
            training:
            ntrees:
            classBand:
            scale: Resolution in meters per pixel.
    """
    trained = ee.Classifier.randomForest(ntrees).train(training, classBand, bands)
    classification = img.select(bands).classify(trained)
    
    return classification.toByte()
