i submitted it but it failed in 6.7s:

s	1	Traceback (most recent call last):
0.8s	2	  File "/kaggle/src/script.py", line 596, in <module>
0.8s	3	    raise SystemExit(main())
0.8s	4	                     ^^^^^^
0.8s	5	  File "/kaggle/src/script.py", line 184, in main
0.8s	6	    cache_dataset = dataset_root(
0.8s	7	                    ^^^^^^^^^^^^^
0.8s	8	  File "/kaggle/src/script.py", line 69, in dataset_root
0.8s	9	    raise KernelError(f"missing or ambiguous dataset {owner}/{slug}")
0.8s	10	KernelError: missing or ambiguous dataset onebitbrain/trackastra-h1-fit105-cache-v1
3.1s	11	/usr/local/lib/python3.12/dist-packages/mistune.py:435: SyntaxWarning: invalid escape sequence '\|'
3.1s	12	  cells[i][c] = re.sub('\\\\\|', '|', cell)
3.3s	13	/usr/local/lib/python3.12/dist-packages/nbconvert/filters/filter_links.py:36: SyntaxWarning: invalid escape sequence '\_'
3.3s	14	  text = re.sub(r'_', '\_', text) # Escape underscores in display text
4.6s	15	[NbConvertApp] Converting notebook **script**.ipynb to html
6.0s	16	[NbConvertApp] Writing 358905 bytes to **results**.html
