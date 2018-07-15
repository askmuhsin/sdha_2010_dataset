### Semantic Description of Human Activities dataset
Recognition of complex human activities from continuous videos, taken in realistic settings. Each video contains several human-human interactions (e.g. hand shaking) occurring sequentially and/or concurrently.     

* Read more about it here : [SDHA 2010](http://cvrc.ece.utexas.edu/SDHA2010/Human_Interaction.html)    


### Usage
`video_to_img.ipynb` contains code to convert video files into file name annotated
image (.png) files, using ffmpeg. The video data can be downloaded from the following section.    

`python inference_from_raw.py ./data/4_3_23.png` will run inference on a LeNet like model.   
`./data/` directory contains some sample data for classification.
The saved model is in `./model` directory.   

`lenet_implemented.ipynb` can be used to train the model.

`get_data.py` is a script to load data and label.    


### Download video data
[Segmented_set_1](http://cvrc.ece.utexas.edu/SDHA2010/videos/competition_1/ut-interaction_segmented_set1.zip)     
[Segmented_set_2](http://cvrc.ece.utexas.edu/SDHA2010/videos/competition_1/ut-interaction_segmented_set2.zip)


#### Cite
'''
@misc{UT-Interaction-Data,
      author = "Ryoo, M. S. and Aggarwal, J. K.",
      title = "{UT}-{I}nteraction {D}ataset, {ICPR} contest on {S}emantic {D}escription of {H}uman {A}ctivities ({SDHA})",
      year = "2010",
      howpublished = "http://cvrc.ece.utexas.edu/SDHA2010/Human\_Interaction.html"
}
'''
