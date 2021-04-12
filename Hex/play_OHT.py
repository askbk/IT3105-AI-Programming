from OHT.BasicClientActor import BasicClientActor


def play_OHT():
    client = BasicClientActor()
    client.connect_to_server()


if __name__ == "__main__":
    play_OHT()
