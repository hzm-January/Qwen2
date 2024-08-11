import re


def clean_query(note):
    # Template remove all repeated whitespaces and more than double newlines
    note = re.sub(r"[ \t]+", " ", note)
    note = re.sub("\n\n\n+", "\n\n", note)
    # Remove all leading and trailing whitespaces
    note = re.sub(r"^[ \t]+", "", note)
    note = re.sub(r"\n[ \t]+", "\n", note)
    note = re.sub(r"[ \t]$", "", note)
    note = re.sub(r"[ \t]\n", "\n", note)
    # Remove whitespaces before colon at the end of the line
    note = re.sub(r"\s*\.$", ".", note)
    note = re.sub(r"\s*\.\n", ".\n", note)
    # Remove repeated dots and the end of the line
    note = re.sub(r"\.+$", ".", note)
    note = re.sub(r"\.+\n", ".\n", note)
    # Remove whitespaces before colon at the end of the line
    note = re.sub(r"\s*\.$", ".", note)
    note = re.sub(r"\s*\.\n", ".\n", note)
    # Template remove all repeated whitespaces and more than double newlines
    note = re.sub(r"[ \t]+", " ", note)
    note = re.sub("\n\n\n+", "\n\n", note)
    # Remove repetitive whitespace colon sequences
    return note
