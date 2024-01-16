cd $ControlAug_DATA_DIR
mkdir voc_fsod
cd voc_fsod
wget https://automl-mm-bench.s3.amazonaws.com/DAD/datasets/PascalVoc_COCOStyle.zip
unzip PascalVoc_COCOStyle.zip
rm PascalVoc_COCOStyle.zip
mv PascalVoc_CocoStyle/* ./
rm -r PascalVoc_CocoStyle
