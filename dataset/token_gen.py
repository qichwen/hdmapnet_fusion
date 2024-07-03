import random  
import string  
  
def generate_random_token(length=32):  
    chars = string.ascii_letters + string.digits  
    return ''.join(random.choice(chars) for _ in range(length))  
    #ca9a282c9e77460f8360f564131a8af5
    #MC9cQyINWjPPWxYFOu3A5fcV6zK2deq7

for _ in range(18):  
    token = generate_random_token()  
    print(token)
