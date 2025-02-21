URL="https://archive.ics.uci.edu/static/public/94/spambase.zip"
DEST_DIR="data"
ZIP_FILE="spambase.zip"
echo "Downloading spambase.zip from $URL..."
wget -O $ZIP_FILE $URL
# Step 2: Create the 'data' directory if it doesn't exist
if [ ! -d "$DEST_DIR" ]; then
   echo "Creating directory: $DEST_DIR"
   mkdir $DEST_DIR
fi
echo "Extracting $ZIP_FILE..."
unzip $ZIP_FILE
if [ -f "spambase.data" ]; then
   if [ -f "${DEST_DIR}/spambase.data" ] ; then
      echo "Removing previous spambase.data file"
      rm $DEST_DIR/spambase.data
   fi
   echo "Moving spambase.data to $DEST_DIR"
   mv spambase.data $DEST_DIR/
else
   echo "Something went wrong"
fi
echo "Cleaning up: Removing $ZIP_FILE"
rm $ZIP_FILE
echo "Download and extraction completed successfully."
rm spambase.DOCUMENTATION spambase.names