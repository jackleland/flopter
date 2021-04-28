
tableau_palette = ['#377eb8', '#ff7f00', '#4daf4a',
                   '#f781bf', '#a65628', '#984ea3',
                   '#999999', '#e41a1c', '#dede00']

cb_palette = ['#4477AA', '#66CCEE', '#228833', '#CCBB44',
              '#EE6677', '#AA3377', '#BBBBBB']

contrast_palette = ['#FFFFFF', '#DDAA33', '#BB5566', '#004488', '#000000']

vibrant_palette = ['#0077BB', '#33BBEE', '#009988', '#EE7733',
                   '#CC3311', '#EE3377', '#BBBBBB']

muted_palette = ['#332288', '#88CCEE', '#44AA99', '#117733', '#999933',
                 '#DDCC77', '#CC6677', '#882255', '#AA4499', '#DDDDDD']

paledark_palette = [
    ['#BBCCEE', '#CCEEFF', '#CCDDAA', '#EEEEBB', '#FFCCCC', '#DDDDDD'],
    ['#222255', '#225555', '#225522', '#666633', '#663333', '#555555'],
]

sunset_div_palette = [
    '#364B9A', '#4A7BB7', '#6EA6CD', '#98CAE1', '#C2E4EF', '#EAECCC',
    '#FEDA8B', '#FDB366', '#F67E4B', '#DD3D2D', '#A50026'
]

palettes = {
    'b': cb_palette,
    'c': contrast_palette,
    'v': vibrant_palette,
    'm': muted_palette,
    'p': paledark_palette[0],
    'd': paledark_palette[1],
    'pd': paledark_palette,
    's': sunset_div_palette,
    'sr': sunset_div_palette[5:],
    'sb': sunset_div_palette[5::-1],
}
