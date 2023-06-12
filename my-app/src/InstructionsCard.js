import { Box, Card, CardBody, ChakraProvider, Text } from '@chakra-ui/react';

const InstructionsCard = ({ data }) => {
  return (
    <ChakraProvider>
      <Box height="55vh" display="flex" alignItems="center" justifyContent="center">
        <Card width="1200px">
          <CardBody>
            <Text fontSize="lg" mb={4}>
            Welcome to the Face Emotion Detector! This application uses a CNN model to determine the emotion of faces. Follow these instructions to use the application:
            <br /><br />
            Look at the generated images on the right side of the screen. Each image represents a face. Study the face and try to determine the corresponding emotion.
            <br /><br />
            Click on the emotion buttons below each image to select the emotion you think best represents the face. After selecting an emotion, click the "Generate New Image" button to generate a new face image.
            <br /><br />
            If you didn't select an emotion before generating a new image, an alert will remind you to choose one. If you selected the correct emotion for an image (e.g., "angry"), the accuracy chart will reflect that.
            <br /><br />
            Use the accuracy chart to track your performance and improve your emotion detection skills.

            </Text>
            <Text>
              
            </Text>
          </CardBody>
        </Card>
      </Box>
    </ChakraProvider>
  );
};

export default InstructionsCard;