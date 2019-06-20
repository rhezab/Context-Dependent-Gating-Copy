from run_models import run_mnist_SI_model_shorter, run_mnist_EWC_model_shorter
import time

# run_mnist_EWC_model_shorter(None, 'iXdG')
# run_mnist_SI_model_shorter(None, 'XdG')

start = time.time()
run_mnist_SI_model_shorter(None, 'iXdG')
end = time.time()
print(end - start)
