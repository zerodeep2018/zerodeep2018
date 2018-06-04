
# coding: utf-8

# 乗算レイヤの実装

# MulLayer

# In[14]:

class MulLayer:
    def __init__(self):
        self.x=None
        self.y=None
    
    def forward(self,x,y):
        self.x=x
        self.y=y
        out=x*y
        
        return out
    
    def backward(self,dout):
        dx=dout*self.y#xとyをひっくり返す
        dy=dout*self.x
        
        return dx,dy


# リンゴの例を実装

# In[15]:

apple=100
apple_num=2
tax=1.1


# In[16]:

#layer
mul_apple_layer=MulLayer()
mul_tax_layer=MulLayer()


# In[17]:

#forward
apple_price=mul_apple_layer.forward(apple,apple_num)
price=mul_tax_layer.forward(apple_price,tax)

print(price)#220


# 各変数に関する微分はbackward()で求める

# In[18]:

#backward
dprice=1
dapple_price,dtax=mul_tax_layer.backward(dprice)
dapple,dapple_num=mul_apple_layer.backward(dapple_price)

print(dapple,dapple_num,dtax)#2.2 110 200


# 加算レイヤの実装

# In[19]:

class AddLayer:
    def __init__(self):
        pass
    
    def forward(self,x,y):
        out=x+y
        return out
    
    def backward(self,dout):
        dx=dout*1
        dy=dout*1
        return dx,dy


# リンゴとみかんの例を実装

# In[20]:

apple=100
apple_num=2
orange=150
orange_num=3
tax=1.1


# In[21]:

#layer
mul_apple_layer=MulLayer()
mul_orange_layer=MulLayer()
add_apple_orange_layer=AddLayer()
mul_tax_layer=MulLayer()


# In[22]:

#forward
apple_price=mul_apple_layer.forward(apple,apple_num)#(1)
orange_price=mul_orange_layer.forward(orange,orange_num)#(2)
all_price=add_apple_orange_layer.forward(apple_price,orange_price)#(3)
price=mul_tax_layer.forward(all_price,tax)#(4)


# In[23]:

#backward
dprice=1
dall_price,dtax=mul_tax_layer.backward(dprice)#(4)
dapple_price,dorange_price=add_apple_orange_layer.backward(dall_price)#(3)
dorange,dorange_num=mul_orange_layer.backward(dorange_price)#(2)
dapple,dapple_num=mul_apple_layer.backward(dapple_price)#(1)

print(price)#715
print(dapple_num,dapple,dorange,dorange_num,dtax)#110 2.2 3.3 165 650

