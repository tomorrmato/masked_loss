In multi-label classification, it is common to encounter missing label issues, where some of the samples do not have a label for one or more of the classes. For example, here is an example of multi-label with lots of missing labels.


| class 1  | class 2  |  class 2  |
|---|---|---|
| +  | -  | ?  |
|  ? | +  | ?  |
|  ? | ?  | -  |

This can occur due to various reasons such as the difficulty in obtaining labels for all classes, the cost of labeling, or the presence of rare classes that may not have enough samples to be labeled. 
There are several ways to handle this issue, such as imputation or using probabilistic models have been proposed, which aim to either estimate the missing labels or incorporate the missing label information in the model, but the estimated missing labels or label imputation generally do not represent the ground truth. In this repo, we propose a general and elegant masking technique that handles missing label in training neural network with just a few lines of loss function change, and weights associated with missing label will not be updated in backpropagation.


Here is an example in multi label cross entropy, with just 3 lines of change in the  ```masked_loss.py```, the loss function now handles missing labels. check out ```test_loss.py``` on how the tests are checked.
```
def compute_masked_cross_entropy_loss(
	pred: np.ndarray, 
	label: np.ndarray, 
	mask: Optional[np.ndarray]=None,
) -> float:
	"""
	compute cross entropy loss with or without mask_tensor, if mask is 
	provided, then entries with 0 will be used to mask loss computation 
	between pred and label
	x
	args:
		pred: predictions array of shape (n,c)
		label: label array of shape (n,c)
		mask: optional mask array of shape (n,c), each entry is either 0 or 1.
			given entry mask[m,n], if mask[m,n]==1, it means the pred[m,n] and label[m,n] can be used for loss calculation,
			otherwise, the entry [m,n] will be replaced with 0 loss in loss_tensor[m,n]  
	
	returns cross entropy loss of float type
	"""
	n = label.shape[0]*label.shape[1]
	loss_tensor = label*np.log(pred)+(1-label)*np.log(1-pred)
	# magic block to handle the missing label 
	if mask is not None:
		loss_tensor *= mask
		n = mask.sum()
	# magic block to handle the missing label 
	ce = -np.sum(tensor_loss)/n
	return ce
```
