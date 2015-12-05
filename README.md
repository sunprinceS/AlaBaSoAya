Alabasoya
=========

##Goal:
compare the difference between Linear LSTM and TreeLSTM , does Linear LSTM can learn the "tree structure"?

##Subtask1:
###Slot1:
	LibSVM to classify OTE from sentence
###Slot2:
	還在想
###Slot3:
	* auto-encoder(Linear LSTM)
	* TreeLSTM
	concat sentence embedding and aspect info to predict polarity

##SubTask2:
	Use Slot1's classifier to choose possible category from "text"
	Use this category and those sentence with it to predict the polarity,collect them to text's {OTE,polarity}

##SubTask3:
	similar to subtask2


##Problem to be solved:
	* How to concat categry and sentence embedding?
	* How to use the target word(relate to category)'s info in sentence embedding ?(concat :p)
	* How to use models in specific domain to predict unseen domain
