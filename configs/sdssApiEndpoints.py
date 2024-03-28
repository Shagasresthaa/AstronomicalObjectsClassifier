# Used for getting the centered image for a given object ID
# Takes following parameters
# ra = Right Ascention (in degrees)
# dec = Declination (in degrees)
# scale = Scale of image (in arsec per pixel)
# height = height of image (in pixels)
# width = width of image (in pixels)
# opt = additional options (optional parameter)
SDSS_IMAGE_CUTOUT_BASE = "http://skyserver.sdss.org/dr16/SkyServerWS/ImgCutout/getjpeg?"

# Used for getting the spectra of a given object
# Takes following parameters
# id = SpecObjId from SpecObj DB view
SDSS_SPECTRA_BASE = "http://skyserver.sdss.org/dr18/en/get/specById.ashx?ID="

# Used for searching the SDSS database using SQL
# Takes following parameters
# cmd = SQL command to be executed on DB
# format = To specify the output format (json, csv, fits, xml, html, etc)

SDSS_OBJ_SQL_SEARCH_BASE = "http://skyserver.sdss.org/dr18/SkyServerWS/SearchTools/SqlSearch?"

# For fetching FITS data from Science Archive Server (SAS)
# Takes following parameters
# plateid
# mjd
# fiberid

SAS_SPEC_FITS_FETCH_BASE = "http://dr18.sdss.org/optical/spectrum/view/data/format=fits?"
