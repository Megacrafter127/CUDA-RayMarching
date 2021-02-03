/usr/local/include/%: %
	pkexec cp $(PWD)/$< /usr/local/include/
/usr/local/lib/%: %
	pkexec cp $(PWD)/$< /usr/local/lib/
/usr/local/bin/%: %
	pkexec cp $(PWD)/$< /usr/local/bin/
/usr/local/sbin/%: %
	pkexec cp $(PWD)/$< /usr/local/sbin/
