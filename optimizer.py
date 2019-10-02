# copy file from mentor yuzhe ma

class FISTA(Optimizer):
	def __init__(self, params, lr=1e-2, gamma=0.6):
		defaults = dict(lr=lr, gamma=gamma)
		super(FISTA, self).__init__(params, defaults)
	
	def step(self, decay=1, closure=None):
		loss = None
		
		if closure is not None:
			loss = closure()
		
		for group in self.param_groups:
			for p in group['params']:
				if p.grad is None:
					continue
				
				grad = p.grad.data
				state = self.state[p]
				
				if 'alpha' not in state:
					print("Initial iteration")
					state['alpha'] = torch.ones_like(p.data) / 2
					state['data'] = p.data
					y = p.data
				else:
					alpha = state['alpha']
					data = state['data']
					state['alpha'] = (1 + (1 + 4 * alpha ** 2).sqrt()) / 2
					y = p.data + ((alpha - 1) / state['alpha']) * (p.data - data)
					state['data'] = p.data
				
				mom = y - group['lr'] * grad
				p.data = self._prox(mom, group['lr'] * group['gamma'])
				
				# no-negative
				p.data = torch.max(p.data, torch.zeros_like(p.data))
		
		return loss
	
	def _prox(self, x, gamma):
		print("gamma:", gamma)
		print("X", torch.max(torch.abs(x)), torch.min(torch.abs(x)))
		y = torch.max(torch.abs(x) - gamma, torch.zeros_like(x))
		
		return torch.sign(x) * y