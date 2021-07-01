from anytree import AsciiStyle, Node, RenderTree
import warnings
import time
import numpy as np

english_months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
ordinal_numbers = ['first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eigth', 'ninth', 'tenth']

def pretty_ast(ast: Node) -> str:
    
    try:
        ast.full_name
        return RenderTree(ast, style=AsciiStyle()).by_attr('full_name')
    except AttributeError:
        return RenderTree(ast, style=AsciiStyle()).by_attr('name')
    
    

def english_length(val, unit):
    
    if unit == '':
        return '%f' % val
    elif unit == 'year':
        if val > 0.75:
            return '%0.1f years' % val
        elif val > 2.0 / 12:
            return '%0.1f months' % (val * 12)
        elif val > 2.0 / 52:
            return '%0.1f weeks' % (val * 52)
        elif val > 2.0 / 365:
            return '%0.1f days' % (val * 365)
        elif val > 2.0 / (24 * 365):
            return '%0.1f hours' % (val * (24 * 365))
        elif val > 2.0 / (60 * 24 * 365):
            return '%0.1f minutes' % (val * (60 * 24 * 365))
        else: 
            return '%0.1f seconds' % (val * (60 * 60 * 24 * 365))
    else:
        warnings.warn('I do not know about this unit of measurement : %s' % unit)
        return 'Unrecognised format'
    
def english_point(val, unit, X):
    #### TODO - be clever about different dimensions?
    unit_range = np.max(X) - np.min(X)
    if unit == '':
        return '%f' % val
    elif unit == 'year':
        time_val = time.gmtime((val - 1970)*365*24*60*60)
        if unit_range > 20:
            return '%4d' % time_val.tm_year
        elif unit_range > 2:
            return '%s %4d' % (english_months[time_val.tm_mon-1], time_val.tm_year)
        elif unit_range > 2.0 / 12:
            return '%02d %s %4d' % (time_val.tm_mday, english_months[time_val.tm_mon-1], time_val.tm_year)
        else:
            return '%02d:%02d:%02d %02d %s %4d' % (time_val.tm_hour, time_val.tm_min, time_val.tm_sec, time_val.tm_mday, english_months[time_val.tm_mon-1], time_val.tm_year)
    else:
        warnings.warn('I do not know about this unit of measurement : %s' % unit)
        return 'Unrecognised format'
    
def to_ordinal(i):
    
    if i < len(ordinal_numbers) -1 :
        return ordinal_numbers[i]
    else:
        return f"{i+1}th"
    