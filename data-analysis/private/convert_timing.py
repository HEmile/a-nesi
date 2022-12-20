with open('timing_dpl.txt', 'r') as f:
    timing_dpl = f.readlines()
    with open ("timing_dpl_converted.csv", "w") as f_csv:
        for line in timing_dpl:
            split = line.split()
            split = list(filter(lambda x: x != '', split))
            write_line = ",".join(split) + "\n"
            f_csv.writelines([write_line])
            print(write_line)

