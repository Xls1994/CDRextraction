# CDRextraction
This is the source code for our paper "Combining knowledge and context representations for chemical-disease relation extraction".
# how to use
Keras> 2.0.1  
tensorflow> 1.0   
python 2.7
## model
* cdr_model.py: The NAM model for CDR extraction.
* customize_layer.py: The attention layer used in NAM.
* cnn.py: A simple demo for CDR extraction.
# CDRextraction v2
We also release our code for the paper "Chemical-induced Disease Relation Extraction with Dependency In-formation and Prior Knowledge"
The model is similar with the NAM, however, we use the CNN to capture the semantic dependency representations (SDR).</br> 
Then, an attention mechanism is employed to learn the importance/weight of the SDR.
## model
* can_model: The CAN model for CDR extraction.

