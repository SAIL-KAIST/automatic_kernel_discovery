
import ast
import logging
from enum import Enum
import numpy as np
from anytree import Node
from gpflow.kernels.linears import Polynomial
from kernel_discovery.description.simplify import simplify
from kernel_discovery.kernel import Periodic, Product, Linear, Constant, RBF, White
from collections import defaultdict
from kernel_discovery.description.transform import ast_to_kernel, ast_to_text
from kernel_discovery.description.utils import english_length, english_point

poly_names = ['linear', 'quadratic', 'cubic', 'quartic', 'quintic']

_NOUN_PHRASES = {
    "periodic": "Periodic function",
    "white" : "Uncorrelated noise",
    "squared_exponential": "Smooth function",
    "constant": "Constant",
    "linear": "Linear function",
    'polynomial': "A polynomial function (of degree `{degree}`)"
}

_DESC_PHRASES = {
    "white": "This component models uncorrelated noise",
    "constant": "This component is constant"
}

_EXTRAP_PHRASES = {
    "white": "This component assumes the uncorrelated noise will continue indefinitely",
    "constant": "This component is assumed to stay constant"
}

_NOUN_PRECEDENCE = {
    'periodic': 0,
    'white': 1,
    'squared_exponential': 2,
    'constant': 3,
    'linear': 4,
    'polynomial': 5
}

_POST_MODIFIERS = {
    'rbf': 'whose shape changes smoothly',
    'periodic': 'modulated by a periodic function',
    'linear': 'with lineary varying amplitude',
    'polynomial': 'with polynomially varying amplitude of degree `{degree}`'
}

def describe(node: Node) -> str:
    
    node = simplify(node)
    return _describe(node)
    
def _describe(node: Node):
    
    if node.is_leaf:
        return _NOUN_PHRASES[node.full_name.lower()] + ';'
    
    if node.name is Product:
        children = list(node.children[:])
        # count the number of linear kernels
        linear_child_count = [child.name for child in children].count(Linear)
        if linear_child_count > 1:
            children = [child for child in children if child.name is not Linear]
            children.append(Node(Polynomial, full_name='Polynomial', degree=linear_child_count))
        
        kernels_by_precedence = sorted(children, key=lambda child: _NOUN_PRECEDENCE[child.full_name.lower()])
        noun, *post_modifiers = kernels_by_precedence
        
        if noun.name is not Polynomial:
            noun_phrase = _NOUN_PHRASES[noun.full_name.lower()]
        else:
            noun_phrase = _NOUN_PHRASES[noun.full_name.lower()].format(degree=noun.degree)
        
        post_modifier_phrases = []
        for post_modifier in post_modifiers:
            if post_modifier.name is Constant:
                continue
            if post_modifier.name is not Polynomial:
                post_modifier_phrases.append(_POST_MODIFIERS[post_modifier.full_name.lower()])
            else:
                post_modifier_phrases.append(_POST_MODIFIERS[post_modifier.full_name.lower()].format(degree=noun.degree))
        
        return " ".join([noun_phrase] + post_modifier_phrases) + ";"
    
    return "\n".join([_describe(child) for child in node.children])

def translate_parametric_window(X, unit='', lin_count=0, exp_count=0, lin_location=None, exp_rate=None, quantity='standard deviation', component='function', qualifier=''):
    
    summary = ''
    description = ''
    extrap_description = ''
    if (lin_count > 0) and (exp_count == 0):
        description += f'The {quantity} of the {component} '
        extrap_description += f'The {quantity} of the {component} is assumed to continue to '
        if lin_count == 1:
            if lin_location < np.min(X):
                summary += f'with {qualifier} linearly increasing {quantity}'
                description += f'increases {qualifier} linearly'
                extrap_description += 'increase %slinearly' % qualifier
            elif lin_location > np.max(X):
                summary += f'with {qualifier}linearly decreasing {quantity}'
                description += f'decreases {qualifier}linearly'
                extrap_description += f'decrease {qualifier}linearly until {english_point(lin_location, unit, X)} after which the {quantity} of the {component} is assumed to start increasing {qualifier}linearly'
            else:
                summary += 'with %s increasing %slinearly away from %s' % (quantity, qualifier, english_point(lin_location, unit, X))
                description += 'increases %slinearly away from %s' % (qualifier, english_point(lin_location, unit, X))
                extrap_description += 'increase %slinearly' % qualifier
        elif lin_count <= len(poly_names):
            summary += 'with %s%sly varying %s' % (qualifier, poly_names[lin_count-1], quantity)
            description += 'varies %s%sly' % (qualifier, poly_names[lin_count-1])
            extrap_description += 'vary %s%sly' % (qualifier, poly_names[lin_count-1])
        else:
            summary += 'with %s given %sby a polynomial of degree %d' % (qualifier, quantity, lin_count)
            description += 'is given %sby a polynomial of degree %d' % (qualifier, lin_count)
            extrap_description += '%s vary according to a polynomial of degree %d' % (qualifier, lin_count)
    elif (exp_count > 0) and (lin_count == 0):
        description += 'The %s of the %s ' % (quantity, component)
        extrap_description += 'The %s of the %s is assumed to continue to ' % (quantity, component)
        if exp_rate > 0:
            summary = 'with exponentially %sincreasing %s' % (qualifier, quantity)
            description += 'increases %sexponentially' % qualifier
            extrap_description += 'increase %sexponentially' % qualifier
        else:
            summary = 'with exponentially %sdecreasing %s' % (qualifier, quantity)
            description += 'decreases %sexponentially' % (qualifier)
            extrap_description += 'decrease %sexponentially' % (qualifier)
    else:
        if exp_rate > 0:
            summary += 'with %s given %sby a product of a polynomial of degree %d and an increasing exponential function' % (quantity, qualifier, lin_count)
            description += 'The %s of the %s is given %sby the product of a polynomial of degree %d and an increasing exponential function' % (quantity, component, qualifier, lin_count)
            extrap_description += 'The %s of the %s is assumed to continue to be given %sby the product of a polynomial of degree %d and an increasing exponential function' % (quantity, component, qualifier, lin_count)
        else:
            summary += 'with %s given %sby a product of a polynomial of degree %d and a decreasing exponential function' % (quantity, qualifier, lin_count)
            description += 'The %s of the %s is given %sby the product of a polynomial of degree %d and a decreasing exponential function' % (quantity, component, qualifier, lin_count)
            extrap_description += 'The %s of the %s is assumed to continue to be given %sby the product of a polynomial of degree %d and a decreasing exponential function' % (quantity, component, qualifier, lin_count)
    return (summary, description, extrap_description)

class Monotonic(Enum):
    INCREASE=1,
    DECREASE=-1

class ProductDesc():
    
    
    
    def __init__(self, prod: Node, x: np.array, monotonic, gradient, unit="") -> None:
        
        self.prod = prod
        self.x = x
        self.domain_range = np.max(x) - np.min(x)
        self.monotic = monotonic
        self.gradient = gradient
        self.unit = unit
        
        # some stats fields
        self.count = defaultdict()
        self.min_period = np.Inf
        self.periodics = []
        
        self.lengthscale = 0
        
        self.linear_location = None
        
        self.unknown_kernel = 0
        self.full_desc = []
        self.extrap_desc = []
        
        if prod.is_leaf:
            kernels = [prod]
        elif prod.name is Product:
            kernels = prod.children
        else:
            raise ValueError("Unrecognized kernel. Either a leaf or a product kernel")
        
        #initialize kernel counting
        for base_kernel in [RBF, Linear, Periodic, White, Constant]:
            self.count[base_kernel] = 0
        
        for k in kernels:
            if k.name is RBF:
                self.count[RBF] += 1
                rbf = ast_to_kernel(k)
                self.lengthscale += rbf.lengthscales.numpy()
            elif k.name is Linear:
                self.count[Linear] += 1
                linear = ast_to_kernel(k)
                self.linear_location = 0.5 # hard code to be remove
                # TODO: handle linear location
            elif k.name is Periodic:
                self.count[Periodic] += 1
                periodic = ast_to_kernel(k)
                self.periodics += [periodic.period.numpy()]
                self.min_period = min(self.periodics)
            elif k.name is White:
                self.count[White] += 1
            elif k.name is Constant:
                self.count[Constant] += 1
            else:
                self.unknown_kernel += 1
        
        self.logger = logging.getLogger(__class__.__name__)
        
    def translate(self):
        
        if self.contain_unknown():
            self.logger.info("")
        elif self.contain_noise():
            self.logger.info("")
        elif self.contain_constant():
            self.logger.info("")
        elif self.contain_smooth():
            self.logger.info("")
        elif self.contain_linear():
            self.logger.info("")
        elif self.contain_periodic():
            self.logger.info("")
        else:
            self.full_desc += ["This simple AI is not capable of describing the component whose python representation is %s' % prod.__repr__()"]
            self.logger.info("")
            
        return self.summary, self.full_desc, self.extrap_desc
    
    def contain_unknown(self):
        
        if self.unknown_kernel > 0:
            self.summary = f"This simple AI is not capable of describing the component whose representation is {ast_to_text(self.prod)}"
            self.full_desc += [f"This simple AI is not capable of describing the component whose representation is {ast_to_text(self.prod)}"]
            self.extrap_desc += [f"This simple AI is not capable of describing the component whose representation is {ast_to_text(self.prod)}"]
            return True
        else:
            return False
    
    def contain_noise(self):
        
        if self.count[White] > 0:
            self.summary = "Uncorrelated noise"
            self.full_desc += ["This component model uncorrelated noise"]
            self.extrap_desc += ["This component assumes the uncorrelated noise will continue indefinitely"]
            if self.count[Linear] > 0:
                summary_, desc, x_desc = translate_parametric_window(self.x, 
                                            unit=self.unit,
                                            lin_count=self.count[Linear],
                                            lin_location=self.linear_location,
                                            component="noise")
                self.summary += f" {summary_}"
                self.full_desc += [desc]
                self.extrap_desc += [x_desc]
            return True
        else:
            return False
        
    def contain_constant(self):
        
        if self.count[RBF] == 0 and self.count[Linear] == 0 and self.count[Periodic]==0:
            self.summary = "A constant"
            self.full_desc += ["This component is constant"]
            self.extrap_desc += ["This component is assumed to stay constant"]
            return True
        else:
            return False
        
    def contain_smooth(self):
        
        if self.count[RBF] > 0 and self.count[Periodic] == 0:
            # long lengthscale
            if self.lengthscale > 0.5 * self.domain_range:
                if self.monotic == Monotonic.INCREASE:
                    self.summary = "A very smooth monotonically increasing function"
                    self.full_desc += ["This component is a very smooth and monotonically increasing function"]
                elif self.monotic == Monotonic.DECREASE:
                    self.summary = "A very smooth monotonically decreasing function"
                    self.full_desc += ["This component is a very smooth and monotonically descreasing function"]
                else:
                    self.summary = "A very smooth function"
                    self.full_desc += ["This component is a very smooth function"]
                self.extrap_desc += ["This component is assumed to continue very smoothly "+
                                     "but is also assumed to be stationary so its distribution will eventually return to the prior"]
            # short lengthscale
            elif self.lengthscale < 0.005 * self.domain_range:
                self.summary = "A rapidly varying smooth function"
                self.full_desc += [f"This component is a rapidly varying but smooth function with a typical lengthscale of {english_length(self.lengthscale, self.unit)}"]
                self.extrap_desc += ["This component is assumed to continue smoothly but its distribution is assumed to quickly return to the prior"]
            else:
                if self.monotic == Monotonic.INCREASE:
                    self.summary = "A smooth monotonically increasing function"
                    self.full_desc += ["This component is a smooth and monotonically increasing function"]
                elif self.monotic == Monotonic.DECREASE:
                    self.summary = "A smooth monotonically decreasing function"
                    self.full_desc += ["This component is a  smooth and monotonically descreasing function"]
                else:
                    self.summary = "A smooth function"
                    self.full_desc += [f"This component is a smooth function with a typical lengthscale of {english_length(self.lengthscale, self.unit)}"]
                self.extrap_desc += ["This component is assumed to continue smoothly "+
                                     "but is also assumed to be stationary so its distribution will eventually return to the prior"]

            self.extrap_desc += [f"The prior distribution places mass on smooth functions with a marginal mean of zero and a typical lengthscale of {english_length(self.lengthscale, self.unit)}"]
            self.extrap_desc+ ["[This is a placeholder for a description of how quickly the posterior will start to resemble the prior]"]
            
            if self.count[Linear] > 0:
                summary_, desc, x_desc = translate_parametric_window(self.x, 
                                            unit=self.unit,
                                            lin_count=self.count[Linear],
                                            lin_location=self.linear_location,
                                            quantity="marginal standard deviation",
                                            component="function")
                self.summary += f" {summary_}"
                self.full_desc += [desc]
                self.extrap_desc += [x_desc]
            
            return True
        else:
            return False
    
    def contain_linear(self):
        
        if self.count[Linear] > 0 and self.count[RBF] == 0 and self.count[Periodic] == 0:
            if self.count[Linear] == 1:
                self.summary = "A linearly increasing function" if self.gradient > 0 else "A linearly decreasing function"
                self.full_desc += ["This component is linearly increasing" if self.gradient > 0 else "This component is linearly decreasing"]
                self.extrap_desc += ["This component is assumed to continue to increase linearly" if self.gradient > 0 else 
                                     "This component is assumed to continue to decrease linearly"]
            elif self.count[Linear] <= len(poly_names):
                name = poly_names[self.count[Linear]-1]
                self.summary = f"A {name} polynomial"
                self.full_desc += [f"This component is a {name} polynomial"]
                self.extrap_desc += [f"This component is assumed to continue as a {name} polynomial"]
            else:
                self.summary = f"A polynomial of degree {self.count[Linear]}"
                self.full_desc += [f"This component is a polynomial of degree {self.count[Linear]}"]
                self.extrap_desc += [f"This component is assumed to continue as a polynomial of degree {self.count[Linear]}"]
                
            return True
        else:
            return False
    
    def contain_periodic(self):
        
        if self.count[Periodic] > 0:
            if self.count[Periodic] == 1:
                one_periodic_only = self.count[Periodic] == 1 and self.count[Linear] == 0 and self.count[RBF] == 0
                one_periodic_with_smooth = self.count[Periodic] == 1 and self.count[Linear] == 0 and self.count[RBF] > 0
                one_periodic_with_linear = self.count[Periodic] == 1 and self.count[Linear] > 0 and self.count[RBF] == 0
                
                if one_periodic_only:
                    summary, desc, x_desc = self.decribe_one_and_only_periodic()
                elif one_periodic_with_smooth:
                    summary, desc, x_desc = self.describe_one_periodic_with_smooth()
                elif one_periodic_with_linear:
                    summary, desc, x_desc = self.describe_one_periodic_with_linear()
                else:
                    summary, desc, x_desc = self.describe_one_periodic_with_mix()
            else:
                summary, desc, x_desc = self.describe_multiple_periodic()

            return True
        else:
            return False
    
    def decribe_one_and_only_periodic(self):
        
        
        
        # TODO: implement this
        return None, None, None
    
    def describe_one_periodic_with_smooth(self):
        
        # TODO: implement this
        return None, None, None
    
    
    def describe_one_periodic_with_linear(self):
        
        return None, None, None
            
    def describe_one_periodic_with_mix(self):
        return None, None, None
    
    def describe_multiple_periodic(self):
        if self.count[RBF] > 0:
            summary = "An approximate product of"
        else:
            summary = "A product of"
        main_desc = "This component is a product of several periodic functions"
        extrap_desc = "This component is assumed to continue as a product of of serveral periodic functions"
        
        desc = [main_desc]
        x_desc = [extrap_desc]
        
        if self.count[RBF] > 0:
            desc += [f"Across periods the shape of this function varies smoothly with a typical lengthscale of {english_length(self.lengthscale, self.unit)}"]
            x_desc +=["Across periods the shape of this function is assumed to continue to vary smoothly but will return to the prior"]
            x_desc +=["The prior is entirely uncertain about the phase of the periodic functions"]
            x_desc +=["Consequently the pointwise posterior will appear to lose its periodicity," +
                      "but this merely reflects the uncertainty in the shape and phase of the functions"]
            x_desc +=["[This is a placeholder for a description of how quickly the posterior will start to resemble the prior]"]
        
        if self.count[Linear] > 0:
            # TODO
            pass
        
        
        return summary, desc, x_desc
    

if __name__ == "__main__":
    
    # unit test
    from kernel_discovery.description.transform import kernel_to_ast
    def test_describe():
        k = (Linear() + RBF()) * Linear()
        ast = kernel_to_ast(k)
        
        desc = describe(ast)
        print(desc)
    
    def test_translate_prod_1():
        k = RBF()
        x = np.linspace(0, 1, 100)[:, None]
        montonic = 1
        gradient = 1.
        prod = kernel_to_ast(k)
        prod_desc = ProductDesc(prod, x, montonic, gradient)
        result = prod_desc.translate()
        print(result)
        
    def test_translate_prod_2():
        """Test white noise"""
        # with linear
        k = White() * Linear()
        x = np.linspace(0, 1, 100)[:, None]
        montonic = 1
        gradient = 1.
        prod = kernel_to_ast(k)
        prod_desc = ProductDesc(prod, x, montonic, gradient)
        result = prod_desc.translate()
        print(result)
        
        # without linear
        k = White()
        prod = kernel_to_ast(k)
        prod_desc = ProductDesc(prod, x, montonic, gradient)
        result = prod_desc.translate()
        print(result)
        
        
    def test_translate_prod_3():
        """Test constant"""
        k = Constant()
        x = np.linspace(0, 1, 100)[:, None]
        montonic = 1
        gradient = 1.
        prod = kernel_to_ast(k)
        prod_desc = ProductDesc(prod, x, montonic, gradient)
        result = prod_desc.translate()
        print(result)
    
    def test_translate_prod_4():
        """Test RBF"""
        # with linear
        k = RBF() * Linear()
        x = np.linspace(0, 1, 100)[:, None]
        montonic = 1
        gradient = 1.
        prod = kernel_to_ast(k)
        prod_desc = ProductDesc(prod, x, montonic, gradient)
        result = prod_desc.translate()
        print(result)
        
        # without linear
        k = RBF()
        prod = kernel_to_ast(k)
        prod_desc = ProductDesc(prod, x, montonic, gradient)
        result = prod_desc.translate()
        print(result)
        
    def test_translate_prod_5():
        
        
        k = Linear()
        x = np.linspace(0, 1, 100)[:, None]
        montonic = 1
        gradient = 1.
        prod = kernel_to_ast(k)
        prod_desc = ProductDesc(prod, x, montonic, gradient)
        result = prod_desc.translate()
        print(result)
        
        k = Linear() * Linear()
        prod = kernel_to_ast(k)
        prod_desc = ProductDesc(prod, x, montonic, gradient)
        result = prod_desc.translate()
        print(result)
        
        k = Linear() * Linear() * Linear()
        prod = kernel_to_ast(k)
        prod_desc = ProductDesc(prod, x, montonic, gradient)
        result = prod_desc.translate()
        print(result)
        
        
    # test_describe()
    # test_translate_prod_1()
    # test_translate_prod_2()
    # test_translate_prod_3()
    # test_translate_prod_4()
    test_translate_prod_5()