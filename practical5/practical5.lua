require 'nngraph'        
x1 = nn.Identity()()    
x2 = nn.Identity()()   
x3 = nn.Identity()()  
linear = nn.Linear(10,10)(x3)    
cmul = nn.CMulTable()({x2,linear})
plus = nn.CAddTable()({cmul,x1})  
network = nn.gModule({x1,x2,x3},{plus})

t1 = torch.randn(10)
t2 = torch.randn(10)
t3 = torch.randn(10)

print(network:forward({t1,t2,t3}))

params  = linear.data.module:parameters()[1]
bias    = linear.data.module:parameters()[2]
print(torch.cmul(t2,(bias+params * t3)) + t1)  

