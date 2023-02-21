
import unittest 
from masked_loss import compute_masked_cross_entropy_loss
import numpy as np 
from typing import Tuple


def reduce_array_by_mask(
	pred: np.ndarray, 
	label: np.ndarray, 
	mask: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
	"""
	input pred and label array contain mask info, which can not be used 
	for computing loss directly, it is important to reduce pred and label
	based on mask array. 

	ex. 
	pred = np.array([[.9,.3,.5],[.1,.4,.6]])
	label = np.array([[1.,0.,0.],[0.,1.,1.]])
	mask = np.array([[1.,1.,0.],[0.,0.,1.]])

	reduce_array_by_mask(pred, label, mask) -> no_mask_pred, no_mask_label
	in this ex, no_mask_pred=np.array([[0.9, 0.3, 0.6]]), no_mask_label=np.array([[1., 0., 1.]])
		
	returns no_mask_pred, no_mask_label arrays that are of shape (1,n) 
		where n is the sum of mask array
	"""
	flattened_pred = pred.reshape(-1)
	flattened_label = label.reshape(-1)
	flattened_mask = mask.reshape(-1)

	no_mask_pred = flattened_pred[flattened_mask != 0].reshape(1,-1)
	no_mask_label = flattened_label[flattened_mask != 0].reshape(1,-1)
	return no_mask_pred, no_mask_label



class TestMaskedLoss(unittest.TestCase):

	def test_loss_equivalent(self):
		# number of runs for testing
		RUNS = 1000
		# random int range
		LOW, HIGH = 1, 200
		# set range to avoid dividing by 0
		EPSILON = 1e-5

		for i in range(RUNS):
			np.random.seed(i)
			# batch size
			n = np.random.randint(low=LOW, high=HIGH)
			# class size
			c = np.random.randint(low=LOW, high=HIGH)

			# generate pred, label and mask arrays
			pred = np.random.uniform(low=EPSILON, high=1-EPSILON, size=(n,c))
			label = np.random.choice([1.,0.], size=(n,c))
			mask = np.random.choice([1.,0.], size=(n,c))

			masked_loss = compute_masked_cross_entropy_loss(pred, label, mask)

			# default cross entropy function does not support missing label/mask, 
			# it is trivial to show that no_mask_pred, no_mask_label are equivalent 
			# to pred, label, mask, so the original_loss can be used for unit test
			no_mask_pred, no_mask_label = reduce_array_by_mask(pred, label, mask)
			original_loss = compute_masked_cross_entropy_loss(no_mask_pred, no_mask_label)
			
			np.testing.assert_almost_equal(masked_loss, original_loss, decimal=5)

if __name__ == '__main__':
	unittest.main()


	