import { ChakraProvider, Box, Center, useColorMode } from "@chakra-ui/react";

function FooterBar() {
  const { colorMode } = useColorMode();

  return (
    <ChakraProvider>
      <Box bg={colorMode === "dark" ? "gray.900" : "gray.200"} p={4} mt={8}>
        <Center>
          <Box fontSize="sm" color={colorMode === "dark" ? "white" : "gray.500"}>
            Created by M. Barron
          </Box>
        </Center>
      </Box>
    </ChakraProvider>
  );
}

export default FooterBar;
