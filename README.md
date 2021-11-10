# Music Genre Classification 
by Eva Garcia Martin

### Folder structure
- code: 
	- python script: `cnn.py`
	- jupyter notebook: `CNN.ipynb`
- models: pre-trained models
- png: figures for the report
- report: report md and pdf

### Report
- report.pdf generated from report.md with the following command:
`pandoc report.md --pdf-engine=xelatex  -o report.pdf`

### Python Script
- For help on usage: `python cnn.py -h`
- Example to train: `python cnn.py train ../data/genres -v -m lenet`
- Example to predict: `python cnn.py predict ../data/genres/  -f ../data/genres/rock/rock.00039.wav -m cnn64 -v`
The script can either train a new CNN from scratch (either a LeNet or a CNN64, for more information please check the report), or predict what is the genre using a pre-trained model given the path to a wav song together with the path to the labels (folders). 

### Data
The data is obtained from: [Download link](http://opihi.cs.uvic.ca/sound/genres.tar.gz) and expected to be saved in a directory called `data` at the same level as the code and models folders. 

### Results
- CNN64: Accuracy: 87% on all training validation and test sets using the CNN64 architecture. Took around 10 hours to run
- LeNet: Accuracy on the training, validations and tests sets, respectively: 94%, 76%, 78% 
