# Google Earth Engine
import ee
ee.Initialize()

# Import Geepy and other libraries
import geepy
import math
import sys

# Get the satellite parameters and variables used
config = geepy.params.configParams('input_classification_v30.json')

# Samples
amostras = ee.FeatureCollection(config.params['samples']['2016v13'])

# Boundary
watersheds = ee.FeatureCollection(config.params['studyArea'])

# Mapped central pivots
pivots = {i: ee.FeatureCollection(config.params['pivots'] + i) for (i) in config.params['years2process']}

# Get satellite bands used
bands = config.params['bandParams']

# Satellite configurations
l5 = ee.ImageCollection(config.params['imgCollection']['lc5']['id']).select(
    config.params["imgCollection"]["lc5"]["bands"],
    config.params["imgCollection"]["lc5"]["bandNames"])

l7 = ee.ImageCollection(config.params['imgCollection']['lc7']['id']).select(
    config.params["imgCollection"]["lc7"]["bands"],
    config.params["imgCollection"]["lc7"]["bandNames"])

l8 = ee.ImageCollection(config.params['imgCollection']['lc8']['id']).select(
    config.params["imgCollection"]["lc8"]["bands"],
    config.params["imgCollection"]["lc8"]["bandNames"])

# DEM
#srtm = ee.Image(config.params['srtm'])
alos = ee.Image(config.params['alos']).select('MED')

# Localities and rivers distance
towns = ee.FeatureCollection(config.params['towns'])
rivers = ee.FeatureCollection(config.params['hidroBDGEx'])
dtown = towns.distance(config.params['radist'])
driver = rivers.distance(config.params['radist'])

# Topographic variables
slope = ee.Terrain.slope(alos)
aspect = ee.Terrain.aspect(alos).divide(180).multiply(math.pi).sin()
hillshade = ee.Terrain.hillshade(alos)

# Night time light world data for all year
ntl30m = {}
for i in config.params['years2process']:
    # VIIRS database
    viirs = ee.Image(config.params['VIIRS'][i]).select('avg_rad').divide(100)
    # Resampling to 30m
    ntl30m[i] = viirs.resample('bilinear').reproject(
        crs = viirs.projection().crs(),
        scale = 30
    )

# Water mask
wmask = driver.lt(300)

# Filtering landsat database
landsat = {}
for year in config.params['years2process']:
    start_d = year + config.params['period']['dry']['start']
    end_d = year + config.params['period']['dry']['end']

    #print("Start-End dry season %s %s" %(start_d, end_d))

    if(int(year) < 2002):
        filtered = l5.filterMetadata('CLOUD_COVER', 'less_than', config.params['cloudCoverThreshold']).filterDate(start_d, end_d).map(geepy.image.maskLandsatSR)
        satellite = 'l5'
    elif(int(year) in (2002, 2011, 2012)):
        filtered = l7.filterMetadata('CLOUD_COVER', 'less_than', config.params['cloudCoverThreshold']).filterDate(start_d, end_d).map(geepy.image.maskLandsatSR)
        satellite = 'l7'
    elif(int(year) > 2002 and int(year) < 2011):
        filtered = l5.filterMetadata('CLOUD_COVER', 'less_than', config.params['cloudCoverThreshold']).filterDate(start_d, end_d).map(geepy.image.maskLandsatSR)
        satellite = 'l5'
    else:
        filtered = l8.filterMetadata('CLOUD_COVER', 'less_than', config.params['cloudCoverThreshold']).filterDate(start_d, end_d).map(geepy.image.maskLandsatSR)
        satellite = 'l8'

    fEdgeRemoved = filtered.map(geepy.image.edgeRemoval).median()

    fEdgeRemoved = geepy.image.img2Band(fEdgeRemoved, ntl30m[year], 'ntl')

    fEdgeRemoved = geepy.image.calcNDBI(fEdgeRemoved)

    #fEdgeRemoved = geepy.image.img2Band(fEdgeRemoved, srtm, 'srtm')
    fEdgeRemoved = geepy.image.img2Band(fEdgeRemoved, wmask, 'wmask')
    fEdgeRemoved = geepy.image.img2Band(fEdgeRemoved, alos, 'alos')
    fEdgeRemoved = geepy.image.img2Band(fEdgeRemoved, slope, 'slope')
    fEdgeRemoved = geepy.image.img2Band(fEdgeRemoved, aspect, 'aspect')
    fEdgeRemoved = geepy.image.img2Band(fEdgeRemoved, hillshade, 'hillshade')

    fEdgeRemoved = geepy.image.img2Band(fEdgeRemoved, dtown, 'dtown')
    fEdgeRemoved = geepy.image.img2Band(fEdgeRemoved, driver, 'driver')

    fEdgeRemoved = geepy.image.calcNDVI(fEdgeRemoved)
    fEdgeRemoved = geepy.image.calcEVI(fEdgeRemoved)
    fEdgeRemoved = geepy.image.calcSAVI(fEdgeRemoved)
    fEdgeRemoved = geepy.image.calcNDWI(fEdgeRemoved)
    fEdgeRemoved = geepy.image.calcMNDWI(fEdgeRemoved)

    fEdgeRemoved = geepy.image.calcRatio(fEdgeRemoved)

    fEdgeRemoved = geepy.image.calcFractions(fEdgeRemoved)
    fEdgeRemoved = geepy.image.calcNDFI(fEdgeRemoved)
    fEdgeRemoved = geepy.image.calcNDFI2(fEdgeRemoved)
    fEdgeRemoved = geepy.image.calcNDFI3(fEdgeRemoved)
    fEdgeRemoved = geepy.image.calcFCI(fEdgeRemoved)
    fEdgeRemoved = geepy.image.calcGVNPV(fEdgeRemoved)
    fEdgeRemoved = geepy.image.calcNPVSOIL(fEdgeRemoved)

    fEdgeRemoved = geepy.image.tassCapTransformation(fEdgeRemoved, satellite)

    ndvithermal = fEdgeRemoved.select('ndvi').divide(fEdgeRemoved.select('thermal'))
    fEdgeRemoved = geepy.image.img2Band(fEdgeRemoved, ndvithermal, 'ndvithermal')

    # Rotina muito pesada para o Google Engine
    #fEdgeRemoved = geepy.image.calcBCI(fEdgeRemoved, watersheds)

    landsat[year] = fEdgeRemoved.clip(watersheds)

    sys.stdout.write("\rProcessing Landsat data: %s" % year)
    sys.stdout.flush()

# Rasterize sample regions
amostragem = geepy.feature.vec2rast(amostras, 'CLASS')

# Classification parameters
n = 4000
classBand = 'CLASS'
cv = [1,2,3,4,5,6,7,8,9,10,11,12]
cp = [n for i in range(len(cv))]
cp[6] = 0    # Agricultura irrigada
#cp[7] = 1000 # Pastagem
cp[9] = 500  # Área urbana

# Get ramdomic samples
samples = geepy.image.randomSamples(amostras, amostragem, n, 369, classBand, cv, cp)
# 1 Formações florestais --------------------------------- #004000
# 2 Formações savânicas ---------------------------------- #77a605
# 3 Mata ciliar ------------------------------------------ #004000
# 4 Grassland -------------------------------------------- #b8af4f
# 5 Agricultura ou pastagem ------------------------------ #f6e6db
# 6 Agricultura de sequeiro ------------------------------ #ffcaff
# 7 Agricultura Irrigada --------------------------------- #ff42f9
# 8 Pastagem  -------------------------------------------- #f4f286
# 9 Corpos d'água ---------------------------------------- #0000ff
# 10 Área urbana/Construções rurais ---------------------- #ff0000
# 11 Solo exposto ---------------------------------------- #77a605
# 12 Rochas ---------------------------------------------- #77a605

# Rasterize central pivots
for year in config.params['years2process']:
    pivots[year] = geepy.feature.vec2rast(pivots[year], 'CLASS').reproject(
        crs = landsat[year].select('nir').projection().crs(),
        scale = 30
    )
    sys.stdout.write("\rProcessing Central Pivots data: %s" % year)
    sys.stdout.flush()

# Change value for pivots
for year in config.params['years2process']:
    pivots[year] = pivots[year].remap([4],[7]).rename(['CLASS'])
    sys.stdout.write("\rProcessing Central Pivots data: %s" % year)
    sys.stdout.flush()

# Accuracy assessment
training = samples.filter(ee.Filter.gt('randCol', 0.5))
validation = samples.filter(ee.Filter.lt('randCol', 0.5))

trained = geepy.image.trainingSamples(landsat['2016'], training)

# Save trained samples to drive
#saveTrained = geepy.feature.send2drive(trained, 'treinamento 2016','treinamento2016v27', 'amostras',' CSV')
#saveTrained.start()

# Randon Forest Classification
classification = {year: geepy.image.randomForest(landsat[year], trained, bands, ntrees=100) for (year) in config.params['years2process']}

# Remap classification
classRemapped = {year: classification[year][0].remap([1,2,3,4,5,6,7,8,9,10,11,12],[1,2,3,4,5,6,6,8,9,10,11,12]).rename('classification'+year) for (year) in config.params['years2process']}

# Replace pivots data in the classification
finalClassification = {year: classRemapped[year].where(pivots[year].select('CLASS'), 7) for year in config.params['years2process']}

# Plot data from GEE
mapa = geepy.maps.geeMap(watersheds, zoom=10)
ano = '2017'
#mapa.addLayer(finalClassification[ano], viz_params=config.params['vizParams']['classification'], name=ano)
mapa.addLayer(landsat[ano], viz_params={'min':0.15,'max':0.4,'bands':'swir1,nir,red'}, name=ano)
mapa.addControls()
mapa.show()

# Region extent
coords = [[[-46.632916356914535, -15.447083782725066],   [-43.13041651144095, -15.447083782725066],   [-43.13041651144095, -10.181249376641395],   [-46.632916356914535, -10.181249376641395],   [-46.632916356914535, -15.447083782725066]]]

# Save to Google Drive
tasks = {year: geepy.image.send2drive(finalClassification[year], coords, 'classification'+year+'v30', 'classification_v30', 30) for year in config.params['years2process']}
for i in tasks.keys():
   [tasks[i].start()]

# Accuracy assessment data
accuracy = {year: geepy.image.accuracyAssessment(classification[year][1]) for year in config.params['years2process']}
accTasks = {year: geepy.feature.send2drive(accuracy[year], 'accuracyAssessment' + year, 'accuracyAssessment_v30', 'accuracyAssessment'+year+'_v30', 'GeoJSON') for year in config.params['years2process']}
for i in accTasks.keys():
    [accTasks[i].start()]

# To asset
# task = {year: geepy.image.send2asset(finalClassification[year],
#                                      coords,
#                                      'classification'+year+'v30',
#                                      'users/fernandompimenta/AIBA/classification_v30',
#                                      30) for year in config.params['years2process']
#        }
#
#
# for i in task.keys():
#     [task[i].start()]
