test_list = [[0,10,102],[7,9,8]]
image_size = 28
depth = 16
print 32 // 4
print image_size // 4 * image_size // 4 * depth

for num_skips, skip_window in [(2, 1), (4, 2)]:
    print num_skips, skip_window

assert 1 % 1 == 0