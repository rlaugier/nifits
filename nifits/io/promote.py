import nifits
"""
	This tool should promote a loaded NIFITS object to the standard NIFITS 1.0
"""
# mysubs = nifits.niio.SUBS_V1
# mykeywords = nifits.niio.NI_NIFITS_DEFAULT_HEADER

def convert_object(mynifits, conversion_list, keywords_list):
    """
    Converts an NIFITS 0.x object into an NIFITS 1.0 object.

    You can get the conversion list and the default header as follows:

	```python
	mysubs = nifits.niio.SUBS_V1
	mykeywords = nifits.niio.NI_NIFITS_DEFAULT_HEADER
	```

    Args:
        mynifits : an `nifits` object
        conversion_list : A list of tuples for the substitution of columns
        keywords_list : A Header object
    Returns:
        newnifits : A new nifits to the 1.0 standard
    """
    from copy import deepcopy
    newnifits = deepcopy(mynifits)
    print("Correcting table column names")
    ext_list = newnifits.extension_objects()
    for anext in ext_list:
        for asub in conversion_list:
            if hasattr(anext, "data_table"):
                if asub[0] in anext.data_table.columns:
                    print(f"Renaming {asub[0]} in {asub[1]}")
                    anext.data_table.rename_column(asub[0], asub[1])
    for akey, anitem in keywords_list.items():
        if akey not in newnifits.header:
            print("Adding ", akey)
            newnifits.header.append((akey, anitem))
        else:
            print(akey, "already exists")
    newnifits.header["HIERARCH NIFITS NI_RMAJ"] = nifits.__standard_version_int__()[0]
    newnifits.header["HIERARCH NIFITS NI_RMIN"] = nifits.__standard_version_int__()[1]
    print(f"Version {newnifits.header["HIERARCH NIFITS NI_RMAJ"]}.{newnifits.header["HIERARCH NIFITS NI_RMIN"]}")
    return newnifits
