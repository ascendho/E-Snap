from redisvl.query import FilterQuery
import redisvl.query
print("Available in redisvl.query:", dir(redisvl.query))
try:
    from redisvl.query import HybridQuery
    print("HybridQuery exists!")
except ImportError:
    print("No HybridQuery imported")
