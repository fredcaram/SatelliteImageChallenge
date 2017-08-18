from agriculture_data_analysis import AgricultureDataAnalysis
from satellite_image_handler import SatelliteImageHandler

stl_image_handler = SatelliteImageHandler()
score_model = stl_image_handler.build_image_model()

pixels_to_remove = stl_image_handler.get_pixels_to_remove()
agr_data_analysis = AgricultureDataAnalysis(pixels_to_remove, score_model, stl_image_handler.satellite_image_size)
agr_data_analysis.general_average_analysis()
agr_data_analysis.analyze_performance_per_item()
