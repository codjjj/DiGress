from src.diffusion.noise_schedule import DiscreteUniformTransition
import torch as t
# uni=DiscreteUniformTransition(2,3,4)
# alpha_bar=t.tensor([[0.5],[0.3]])
# device = "cuda:0"
# Q_t=uni.get_Qt_bar(alpha_bar_t=alpha_bar,device=device)
a=t.ones((5))
b=t.ones((5,3))
print(t.matmul(a,b))