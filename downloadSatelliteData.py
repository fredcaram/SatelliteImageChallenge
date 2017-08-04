import urllib.request

country = "BRA"
initialYear = 2016
initialWeek = 1
finalYear = 2017
finalWeek = 52

for year in range(initialYear, finalYear + 1):
    for week in range(initialWeek, finalWeek + 1):
        url = "https://www.star.nesdis.noaa.gov/smcd/emb/vci/VH/image_country_G04L01.php?country={0}&source=AVHRR-VHP&type=SMN&mask=1&title=SMN&zoom=-1&week={1},{2}".format(country, year, week)
        fileImage = urllib.request.urlopen(url)
        imagePath = 'satelliteImages\map_{0}_y{1}_w{2}.png'.format(country, year, week)
        with open(imagePath,'wb') as output:
            output.write(fileImage.read())