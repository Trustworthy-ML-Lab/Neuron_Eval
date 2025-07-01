curl -L -o CUB_200_2011.tgz https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz
tar -xf CUB_200_2011.tgz -C data
del CUB_200_2011.tgz
cd data
gdown https://drive.google.com/uc?id=12oqqbALrp9b0AYxIGNsq_gvI3nCJD9EF
tar -xf CUB_processed.zip -C .
del CUB_processed.zip
cd ..