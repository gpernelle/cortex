import ipyparallel as ipp

#px cd '/Users/GP1514/Dropbox/ICL-2014/Code/C-code/cortex/notebooks'
c = ipp.Client(profile='cluster')

print(c.ids) 
