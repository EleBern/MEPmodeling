
from MEPmodel_bio import MEPmodel_bio

def objective_function(p, ref):
    """
    Objective function: runs the biological MEP model and returns the residual.
    """
    _, ref_updated = MEPmodel_bio(p, ref)
    error = ref_updated['error']
    return error, ref_updated