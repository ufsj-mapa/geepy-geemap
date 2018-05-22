"""
General Feature functions
"""

import ee
ee.Initialize()

def getTiles(feature, tile, attr):
    """
    getTiles help
    
    Args:
        feature:
        tile:
        attr:
    """
    return [tile, feature.filter(ee.Filter.eq(attr, tile))]

def featureArea(feature):
    """
    areaHA help
    
    Args:
        feature:
    """
    return feature.set({'areaHA':feature.geometry().area().divide(100*100)})

def setGeometry(feature):
    """
    Returns the feature, with the geometry replaced by the specified geometry.

    Args:
        feature: Input feature (geometry or fusion table).
    """
    lon = feature.get("longitude")
    lat = feature.get("latitude")
    geom = ee.Algorithms.GeometryConstructors.Point([lon, lat])
    return feature.setGeometry(geom)

def send2drive(ftr, desc, driveFolder, fnp, ff):
    """
    Creates a batch task to export a feature to Google Drive.
    
    Args:
        ftr: Input feature (geometry, table)
        desc: A human-readable name of the task.
        driveFolder: The Google Drive Folder that the export will reside in.
        fnp: The file name prefix.
        ff: The output file format (CSV, GeoJSON, KML or KMZ).
    """
    task2Drive = ee.batch.Export.table.toDrive(
        collection = ftr,
        folder = driveFolder,
        description = desc,
        fileNamePrefix = fnp,
        fileFormat = ff,
    )
    return task2Drive

def vec2rast(feature, className = 'CLASS',):
    """
        Rasterizes a feature collection using an attribute field.
        
        Args:
            feature: Input feature (geometry or fusion table).
            className: An attribute field on the features to be used for rasterize.
    """
    img = feature.filter(ee.Filter.neq(className, None)).reduceToImage(
        properties = [className],
        reducer = ee.Reducer.first()
    )
    
    img = img.addBands(ee.Image.pixelLonLat())
    
    return img.rename([className, "longitude", "latitude"]).cast({className: "int8"})
