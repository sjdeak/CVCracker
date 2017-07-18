def set_grid_stretchable(win):
    num_of_cols, num_of_rows = win.grid_size()

    for i in range(num_of_rows):
        win.rowconfigure(i, weight=1)
    for j in range(num_of_cols):
        win.columnconfigure(j, weight=1)