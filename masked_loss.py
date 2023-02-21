

import numpy as np 
from typing import Optional 


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
	


