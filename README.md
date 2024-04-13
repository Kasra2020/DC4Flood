# DC4Flood

# DC4Flood: A deep clustering framework for rapid flood detection using Sentinel-1 SAR imagery

Severe flood losses have been on the rise, and this trend is expected to become increasingly prevalent in the future due to climate and socio-economic changes. Swiftly identifying flooded areas is crucial for mitigating socio-economic losses and facilitating effective recovery. Synthetic Aperture Radar (SAR) sensors are operational in all-weather, day-and-night conditions and offer a rapid, accurate, and cost-effective means of obtaining information for quick flood mapping. However, the complex nature of SAR images, such as speckle noise, coupled with the often absence of training/labeled samples, presents significant challenges in their processing procedures. To alleviate such hindrances, we can benefit from unsupervised classification approaches (also known as clustering). Clustering methods offer valuable insights into newly acquired datasets without the need for training or labeled samples. However, traditional clustering approaches are predominantly linear-based and overlook the spatial information of neighboring pixels during analysis. Thus, to attenuate these challenges, we propose a deep learning (DL)-based clustering approach for flood detection (DC4Flood) using SAR images. The primary advantage of DC4Flood over existing DL-based clustering approaches lies in its ability to capture multi-scale spatial information. This is achieved by utilizing multiple dilated convolutions with varying dilation rates and subsequently fusing the extracted multi-scale information to effectively and efficiently analyze SAR images in an unsupervised manner. Extensive experiments conducted on SAR images from six different flood events demonstrate the effectiveness of the proposed DC4Flood.

If you use this code please cite the following paper, K. Rafiezadehshahi, A. Camero, J. Eudaric, and H. Kreibich, "DC4Flood: A deep clustering framework for rapid flood detection using Sentinel-1 SAR imagery", in IEEE Geoscience Remote Sensing Letters. Upon publication, the DOI of the paper will be available.
