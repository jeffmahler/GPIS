"""
Functions to import for sdf class/maybe other things. Avoiding potential clutter

Author: Sahaana Suri
"""
def has_p_and_n(elems):
    """ 
    Function to determine if a np array has both negative and positive values
    Params: 
        elems: np array of numbers
    Returns: 
        (bool): True if elems has both negative and positive numbers, False otherwise
    """
    return (elems>0).any() and (elems<0).any()
